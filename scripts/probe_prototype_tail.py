#!/usr/bin/env python3
"""Offline mechanism probe: dropping history vs count-weighted block prototypes.

No online method or performance claim. Index-time post-RoPE K/V moments would
be required in deployment; selected tokens must not also enter the tail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_complete_attention_capture import construct_routes, validate_capture
from adapters.longlive_sparse.route_plan import map_union_coordinates
from adapters.longlive_sparse.offline_eval import dense_history_attention, output_error_metrics


def weighted_attention(q, k, v, multiplicity, chunk=128):
    outputs = []
    bias = multiplicity.float().log()
    for start in range(0, q.shape[0], chunk):
        logits = q[start:start+chunk].float() @ k.float().T * q.shape[-1]**-.5 + bias
        outputs.append(torch.softmax(logits, dim=-1) @ v.float())
    return torch.cat(outputs)


def selected_raw_indices(plan, frame_ids, token_ids):
    if plan.groups != 1:
        raise ValueError('first tail probe requires shared per-head routes')
    return map_union_coordinates(plan, frame_ids, token_ids)


def tail_output(capture, selections, *, block_tokens=64, device='cuda'):
    q, k, v, ek, ev = [capture[name].to(device) for name in ('query', 'key', 'value', 'exact_key', 'exact_value')]
    output = torch.empty_like(q, dtype=torch.float32)
    raw_count = prototype_count = represented = 0
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            chosen = selections[b][h].to(device)
            selected = torch.zeros(k.shape[1], dtype=torch.bool, device=device)
            selected[chosen] = True
            remaining = ~selected
            keys = [ek[b, :, h].float(), k[b, chosen, h].float()]
            values = [ev[b, :, h].float(), v[b, chosen, h].float()]
            counts = [torch.ones(ek.shape[1], device=device), torch.ones(chosen.numel(), device=device)]
            frames = capture['frame_ids'][b, h].to(device)
            tokens = capture['token_ids'][b, h].to(device)
            # Only historical committed features enter these moments. The
            # teacher output is not used to choose any raw token or group.
            for frame in torch.unique(frames, sorted=True).tolist():
                frame_mask = frames == frame
                length = int(tokens[frame_mask].max())+1
                for start in range(0, length, block_tokens):
                    mask = remaining & frame_mask & (tokens >= start) & (tokens < start+block_tokens)
                    count = int(mask.sum())
                    if count:
                        keys.append(k[b, mask, h].float().mean(0, keepdim=True))
                        values.append(v[b, mask, h].float().mean(0, keepdim=True))
                        counts.append(torch.tensor([count], device=device, dtype=torch.float32))
                        prototype_count += 1
                        represented += count
            output[b, :, h] = weighted_attention(q[b, :, h], torch.cat(keys), torch.cat(values), torch.cat(counts))
            raw_count += chosen.numel()
    token_bytes = 2*q.shape[-1]*q.element_size()
    return output, dict(raw_history_tokens=raw_count, prototype_tokens=prototype_count,
        approximate_original_tokens=represented, raw_payload_bytes=raw_count*token_bytes,
        hypothetical_BF16_prototype_payload_bytes=prototype_count*token_bytes,
        prototype_compute_dtype='FP32_means_for_mechanism_probe',
        prototype_counts_metadata_bytes=4*prototype_count,
        raw_plus_approximate_coverage=raw_count+represented,
        physical_transfer_not_measured=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--kind', choices=('motion', 'state'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layers', default='0,9,19,29')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU replay required')
    audit = json.loads((args.workspace/'results/metrics/matched_trajectory_capture_be00491/trajectory_audit.json').read_text())
    case = next(c for c in audit['cases'] if c['method'] == 'rag_dense' and c['prompt'] == 'calibration_'+args.kind)
    report = dict(status='running', scope='offline_count_weighted_prototype_tail_mechanism',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), gpu=torch.cuda.get_device_name(),
        selection_uses_teacher_output=False, speed_claim=False, online_implementation=False, cases=[])
    try:
        for layer in map(int, args.layers.split(',')):
            path = Path(case['capture_dir'])/f'layer{layer:02d}_start00046800_pass00.pt'
            capture = torch.load(path, map_location='cpu', weights_only=True)
            validate_capture(capture)
            route = construct_routes(capture, device='cuda', value_candidates=())['legacy_cap25']
            indices = selected_raw_indices(route, capture['frame_ids'], capture['token_ids'])
            selections = [[indices[b, h, :int(route.group_history_counts[b, h, 0])].cpu()
                           for h in range(indices.shape[1])] for b in range(indices.shape[0])]
            no_raw = [[torch.empty(0, dtype=torch.long) for _ in heads] for heads in selections]
            generator = torch.Generator().manual_seed(20260907)
            random_raw = [[torch.randperm(capture['key'].shape[1], generator=generator)[:len(row)].sort().values
                           for row in heads] for heads in selections]
            # Freeze all selections BEFORE teacher output evaluation.
            methods = dict(prototype_only=no_raw, legacy25_plus_tail=selections, random25_plus_tail=random_raw)
            q, k, v, ek, ev = [capture[name].cuda() for name in ('query', 'key', 'value', 'exact_key', 'exact_value')]
            reference = dense_history_attention(q, torch.cat((ek, k), 1), torch.cat((ev, v), 1))
            from adapters.longlive_sparse.offline_eval import routed_history_attention
            dropped = routed_history_attention(q, k, v, capture['frame_ids'], capture['token_ids'], route,
                                               exact_key=ek, exact_value=ev)
            rows = {'legacy25_drop': {'output_error': output_error_metrics(reference, dropped)}}
            for name, selected in methods.items():
                output, accounting = tail_output(capture, selected)
                assert accounting['raw_plus_approximate_coverage'] == k.shape[0]*k.shape[1]*k.shape[2]
                rows[name] = dict(output_error=output_error_metrics(reference, output), accounting=accounting)
            result = dict(layer=layer, prompt=case['prompt'], capture=str(path), route_sha=route.digest(), rows=rows,
                post_RoPE_prototypes_reconstructed_offline=True, no_future_video_or_dense_output_used_in_raw_selection=True)
            report['cases'].append(result)
            (args.output/f'layer{layer:02d}.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'layer':layer, 'kind':args.kind, 'relative_l2':{n:r['output_error']['relative_l2'] for n,r in rows.items()}}),flush=True)
            del capture, q, k, v, ek, ev, reference, dropped, output
        report['status'] = 'pass'
    except BaseException:
        report.update(status='fail', traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
