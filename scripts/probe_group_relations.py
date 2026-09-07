#!/usr/bin/env python3
"""Complete-capture grouping/admission audit, never a video-quality claim.

All online-legal routes are frozen before computing teacher outputs/weights.
Matched pair budgets need not have matched transfer unions; both are reported.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.group_relations import summarize_groups, build_group_relation_route
from adapters.longlive_sparse.offline_eval import dense_history_attention, routed_history_attention, output_error_metrics
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from scripts.evaluate_complete_attention_capture import validate_capture, retained_probability_mass


def build_routes(capture, device):
    validate_capture(capture)
    params = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    config = SparseHistoryConfig(method='transfer_vaware_hybrid_history', history_density=.25, method_params=params)
    archive = HistoryArchive(config, spatial_height=capture['spatial_height'], spatial_width=capture['spatial_width'])
    frame_ids, token_ids = capture['frame_ids'].long().cpu(), capture['token_ids'].long().cpu()
    frames = list(dict.fromkeys(frame_ids[0, 0].tolist()))
    actual = capture.get('actual_online_context')
    if actual and actual.get('metadata', {}).get('key_prototype_space', 'unrotated') != 'unrotated':
        raise ValueError('raw-proxy experiment cannot consume aligned prototype captures')
    for frame in frames:
        mask = frame_ids[0, 0] == frame
        order = token_ids[0, 0, mask].argsort()
        key = capture['key_unrotated'][:, mask][:, order].cpu()
        value = capture['value'][:, mask][:, order].cpu()
        archive.index_frame(0, frame, key.to(device), value, storage_k=key, storage_v=value)
        if actual:
            block_mask = actual['block_frame_ids'] == frame
            old = archive._layers[0][frame]
            archive._layers[0][frame] = replace(old, block_centroids=actual['key_prototypes'][:, :, block_mask],
                block_value_centroids=actual['value_prototypes'][:, :, block_mask])
    raw_q = capture['query_unrotated'].to(device)
    original_summary = summarize_query_for_pretransfer(raw_q, 64)
    routes = {'legacy_final25': archive.route_indexed(0, original_summary, frames, exact_k_tokens=capture['exact_key'].shape[1])}
    construction = {}
    for grouping in ('random_balanced', 'spatial_quadrants', 'query_features'):
        summary = summarize_groups(raw_q, grouping=grouping, spatial_height=capture['spatial_height'],
                                   spatial_width=capture['spatial_width'])
        context = archive.online_routing_context(0, summary, frames)
        construction[grouping] = {'q_summary_s': summary.q_summary_s, 'D2H_s': summary.d2h_s,
            'summary_bytes_including_labels': summary.summary_bytes, 'group_sizes': summary.query_group_sizes.tolist()}
        for density in (.25, .5):
            for admission in ('shared', 'per_group'):
                name = f'{grouping}_{admission}_{density}'
                started = time.perf_counter()
                routes[name] = build_group_relation_route(context, summary.query_labels, frame_ids, token_ids,
                    exact_tokens=capture['exact_key'].shape[1], grouping=grouping, admission=admission, density=density)
                construction[name] = {'CPU_route_s': time.perf_counter()-started}
    return routes, construction


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--kind', choices=('motion', 'state'), required=True)
    p.add_argument('--layers', default='0,29')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU teacher replay required')
    audit = json.loads((args.workspace/'results/metrics/matched_trajectory_capture_be00491/trajectory_audit.json').read_text())
    case = next(c for c in audit['cases'] if c['method'] == 'rag_dense' and c['prompt'] == 'calibration_' + args.kind)
    result = dict(status='running', prompt=case['prompt'], gpu=torch.cuda.get_device_name(),
        source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/probe_group_relations.py', 'adapters/longlive_sparse/group_relations.py')},
        scope='offline_complete_attention_grouping_probe', formal_promotion=False, cases=[])
    try:
        for layer in map(int, args.layers.split(',')):
            path = Path(case['capture_dir'])/f'layer{layer:02d}_start00046800_pass00.pt'
            capture = torch.load(path, map_location='cpu', weights_only=True)
            routes, construction = build_routes(capture, 'cuda')
            frozen_shas = {name: route.digest() for name, route in routes.items()}
            # Teacher context becomes available only AFTER routes are built.
            q, k, v, ek, ev = [capture[name].cuda() for name in ('query', 'key', 'value', 'exact_key', 'exact_value')]
            teacher = dense_history_attention(q, torch.cat((ek, k), 1), torch.cat((ev, v), 1))
            mass = retained_probability_mass(capture, routes, device='cuda')
            rows = {}
            for name, route in routes.items():
                output = routed_history_attention(q, k, v, capture['frame_ids'], capture['token_ids'], route,
                    exact_key=ek, exact_value=ev)
                if frozen_shas[name] != route.digest():
                    raise RuntimeError('route changed after teacher became available')
                rows[name] = dict(output_error=output_error_metrics(teacher, output), probability_mass=mass[name],
                    route_sha=route.digest(), history_pair_density=route.history_pair_density,
                    unique_transfer_density=route.history_transfer_density,
                    global_pair_density=route.global_executed_density,
                    unique_payload_bytes=route.unique_history_tokens*2*q.shape[-1]*q.element_size(),
                    physical_compact_bytes=route.union_frame_ids.numel()*2*q.shape[-1]*q.element_size(),
                    executor=route.grouped_executor_storage(head_dim=q.shape[-1], element_size=q.element_size()))
            report = dict(status='pass', layer=layer, capture=str(path), capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                all_routes_built_before_teacher=True, construction=construction, rows=rows,
                full_exact_context_included=True, unbounded_union_is_not_equal_byte_comparison=True)
            result['cases'].append(report)
            (args.output/f'layer{layer:02d}.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'layer': layer, 'prompt': case['prompt'], 'metrics': {name: {
                'relative_l2': row['output_error']['relative_l2'], 'history_pairs': row['history_pair_density'],
                'transfer': row['unique_transfer_density']} for name, row in rows.items()}}), flush=True)
            del capture, q, k, v, ek, ev, teacher, output
        result['status'] = 'pass'
    except BaseException:
        result.update(status='fail', traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
