"""Bounded full-H3 cache correctness/cost probe; no benchmark quality claim."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_shared_stage1 import shared_modules, TransformerCallProxy
from h3_shared_kv_heads import require_same_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    args = ap.parse_args()
    request = json.loads(Path(args.request).read_text())
    torch.set_num_threads(2)
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20), device_id=device)
    out = Path(os.environ['INFER_OUTPUT_DIR'])
    try:
        world=dist.get_world_size()
        if world!=request.get('head_shards',8) or world not in [4,8]:
            raise ValueError('Use the declared same-video four/eight-rank group')
        code, manifest = shared_modules()
        core = code['h3_fixed_kv_stage1']
        order_probe=None
        if request.get('kv_order_numeric_diagnostic') is True:
            from h3_kv_order_numeric import install
            order_probe,_=install(core)
        data = code['h3_stage1_abc_data']
        runtime = code['h3_fixed_kv_runtime']
        require_same_request(hashlib.sha256(Path(args.request).read_bytes()).hexdigest())
        session = load_session(request['weights_spec'], device, load_decoders=False)
        proxy = TransformerCallProxy(session['model'])
        source = json.loads(Path(request['source_file']).read_text())
        blocks = data.Reader().sample(source, request['window'])
        bank = core.VideoBank(detach=True)
        gen = torch.Generator(device=device).manual_seed(request['seed'])
        def rebuild_history():
            rebuilt = core.VideoBank(detach=True)
            with torch.inference_mode():
                for b in blocks[:-1]:
                    rebuilt.evict(b['past'])
                    packed = data.pack([b], device, gen, clean_only=True)
                    ctx = core.Context(packed['context'].splits, False, rebuilt.snapshot(), rebuilt)
                    core.forward(proxy, packed['inputs'], ctx)
                rebuilt.evict(blocks[-1]['past'])
            return rebuilt
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.inference_mode():
            bank = rebuild_history()
            torch.cuda.synchronize()
            prefill_seconds = time.monotonic()-tick
            current = blocks[-1]
            bank.evict(current['past'])
            history_tokens = sum(x.key.shape[2] for x in bank.layers[0])
            if history_tokens < 50*1008:
                raise ValueError('Probe does not reach a representative full history window')
            complete = data.pack(blocks, device, torch.Generator(device=device).manual_seed(request['seed']))
            torch.cuda.synchronize()
            tick = time.monotonic()
            full_v, full_a = core.forward(proxy, complete['inputs'], complete['context'])
            torch.cuda.synchronize()
            reference_seconds = time.monotonic()-tick
            last = complete['outputs'][-1]
            a,b = last['video_slice']
            expected_v = full_v[:,a:b].clone()
            aa,ab = last['audio_slice']
            expected_a = full_a[:,aa:ab].clone()
            cached = data.pack([current], device, torch.Generator(device=device).manual_seed(request['seed']), include_clean=False)
            ci, fi = cached['inputs'], complete['inputs']
            nv,na,nt = current['video'].shape[0],current['audio'].shape[0],current['prompt'].shape[0]
            ci['hidden_states'] = fi['hidden_states'][:,-nv:].clone()
            ci['audio_hidden_states'] = fi['audio_hidden_states'][:,-na:].clone()
            ci['encoder_hidden_states'] = fi['encoder_hidden_states'][:,-nt:].clone()
            ids = torch.cat([fi['text_indices'][-nt:],fi['video_indices'][-nv:],fi['audio_indices'][-na:]])
            ci['timestep'] = fi['timestep'].clone()
            ci['timestep_indices'] = fi['timestep_indices'][ids].clone()
            ci['position_ids'] = fi['position_ids'][ids].clone()
            ctx = core.Context(cached['context'].splits,False,bank.snapshot())
            torch.cuda.synchronize()
            tick = time.monotonic()
            actual_v, actual_a = core.forward(proxy, ci, ctx)
            torch.cuda.synchronize()
            cached_seconds = time.monotonic()-tick
            def error(actual, expected):
                return dict(relative_l2=float(torch.linalg.vector_norm(actual.float()-expected.float()) /
                            torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)),
                            max_abs=float((actual.float()-expected.float()).abs().max()))
            packed_errors = dict(video=error(actual_v,expected_v),audio=error(actual_a,expected_a))
            del complete, full_v, full_a, expected_v, expected_a
            torch.cuda.empty_cache()
            # Same-mask SAME-SHAPE oracle: re-encode every clean block in its
            # original commit order, then evaluate exactly the same current
            # inputs. Packed TF uses different GEMM/reduction shapes and stays
            # an explicit auxiliary training/inference numerical comparison.
            torch.cuda.synchronize()
            tick = time.monotonic()
            rebuilt = rebuild_history()
            cache_exact = all(torch.equal(a.key,b.key) and torch.equal(a.value,b.value)
                for layer in bank.layers for a,b in zip(bank.layers[layer],rebuilt.layers[layer]))
            if set(bank.layers)!=set(rebuilt.layers) or any(
                [x.block for x in bank.layers[layer]]!=[x.block for x in rebuilt.layers[layer]] for layer in bank.layers):
                raise RuntimeError('Rebuilt reference has a different committed history layout')
            reference_v,reference_a=core.forward(proxy,ci,core.Context(cached['context'].splits,False,rebuilt.snapshot()))
            torch.cuda.synchronize()
            sequential_reference_seconds=time.monotonic()-tick
            errors=dict(video=error(actual_v,reference_v),audio=error(actual_a,reference_a))
            passed=cache_exact and all(v['relative_l2']<=request['same_shape_relative_l2_tolerance'] and
                v['max_abs']<=request['same_shape_max_abs_tolerance'] for v in errors.values())
            del rebuilt, reference_v, reference_a, cached, actual_v, actual_a
            torch.cuda.empty_cache()
            if not passed:
                raise RuntimeError('Same-shape recomputation versus committed cache failed: '+json.dumps(errors))
            stream = runtime.FixedKVStream(proxy,'A')
            stream.bank = bank
            stream.blocks = {b['block']:dict(start=b['source_start'],end=b['source_end']) for b in blocks[:-1]}
            stream.last_block = blocks[-2]['block']
            torch.cuda.synchronize()
            tick = time.monotonic()
            video, audio, stats = stream.generate(current, gen, device, 50)
            torch.cuda.synchronize()
            chunk_seconds = time.monotonic()-tick
        report = dict(status='PASS', rank=rank, world=world, GPU=torch.cuda.get_device_name(device),
            shared_source=manifest['bundle_sha256'], model_load_seconds=session['load_seconds'],
            clean_history_prefill_seconds=prefill_seconds, history_tokens=history_tokens,
            cached_KV_bytes_per_rank=bank.bytes,
            same_new_rules_BF16_reference_error=errors,
            same_shape_reference_recomputes_all_clean_history=True,
            committed_KV_rebuilt_bit_exact=cache_exact,
            same_shape_relative_l2_tolerance=request['same_shape_relative_l2_tolerance'],
            same_shape_max_abs_tolerance=request['same_shape_max_abs_tolerance'],
            packed_TF_vs_cache_auxiliary_error=packed_errors,
            packed_TF_numeric_gap_not_silently_claimed_equal=True,
            same_shape_recomputed_history_plus_one_NFE_seconds=sequential_reference_seconds,
            one_NFE_recomputed_seconds=reference_seconds, one_NFE_cached_seconds=cached_seconds,
            full_50_NFE_plus_clean_commit_chunk_seconds=chunk_seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            total_memory_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
            clean_commit_forwards=stats['clean_commit_forwards'],
            old_video_queries_recomputed=stats['old_video_queries_recomputed'],
            generated_benchmark_videos=0, formal_memory_quality_claim=False,
            source_is_training_reader_capacity_probe=True,
            diagnostic_KV_concat_order='cached_past_first' if order_probe is not None else 'unmodified_shared_order',
            diagnostic_reordered_cat_calls=order_probe.reordered_calls if order_probe is not None else 0,
            projected_85_chunk_60s_seconds_with_load_and_15min_delivery_reserve=
                session['load_seconds']+85*chunk_seconds+900)
        (out/f'rank{rank}_PROBE.json').write_text(json.dumps(report,indent=2)+'\n')
        if rank == 0:
            print(json.dumps(report),flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
