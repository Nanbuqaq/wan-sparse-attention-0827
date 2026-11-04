"""Inference for the original native DMD contract, distinct from Stage1B.

Uses the training sink/recent processor and re-noising algebra verbatim.
Production inference always finishes four steps; random training exits are
never sampled here. No old latent is evaluated after clean KV commitment.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

CONTRACT = 'chunk_causal_sink3_recent9_native_text_stochastic4_DMD_v1'
FROZEN = Path(__file__).resolve().parent / 'native_dmd_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate(request, verify_checkpoints=False, *, contract=CONTRACT, schema='h3_native_dmd_eval_v1'):
    if (request.get('schema') != schema or request.get('attention_contract') != contract
            or request.get('NFE') != 4 or request.get('chunks') not in (4, 84, 85)
            or request.get('history_recomputation_allowed') is not False):
        raise ValueError('Explicit native DMD fixed-KV four-step request required')
    manifest = json.loads((FROZEN / 'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(FROZEN / name) != digest: raise ValueError('Frozen training source differs: ' + name)
    for item in request['checkpoints']:
        spec = item['weights_spec']
        if spec['attention_contract'] != contract or spec['kind'] != 'full' or not spec['diagnostic_only']:
            raise ValueError('Unsupported model spec')
        payload = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        actual_contract=payload.get('attention_contract')
        initialization_control=(contract==CONTRACT
            and spec.get('comparison_track')=='initialization_weights_native_dmd_inference_rules'
            and actual_contract=='chunk_causal_local12_readonly_text_no_audio_history_full_sequence_ODE_v2'
            and spec.get('initialization_source_attention_contract')==actual_contract)
        if (payload.get('schema') != 'h3_full_dit_inference_v1'
                or (actual_contract != contract and not initialization_control) or payload['step'] != item['step']):
            raise ValueError('Export header and request disagree')
        loras=[k for k in payload['transformer'] if '.lora_' in k]
        if contract == CONTRACT and loras:
            raise ValueError('Native dense full checkpoint expected')
        if contract == 'STAGE1B_CURRENT_ONLY_V1' and (len(loras)!=400 or spec.get('adapter_layout')!='all_attention_rank8_alpha16'):
            raise ValueError('Explicit Stage1B all-attention LoRA export required')
        if any('.hsa_' in k for k in payload['transformer']):raise ValueError('HSA export requires a separate worker')
        del payload
        if verify_checkpoints and sha(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('Checkpoint content hash mismatch')
    for prompt in request['prompts']:
        if sha(prompt['path']) != prompt['file_sha256']: raise ValueError('Prompt file changed')
        emb = torch.load(prompt['path'], map_location='cpu', weights_only=True)
        if emb['prompt'] != prompt['text'] or tensor_sha(emb['prompt_embeds']) != prompt['tensor_sha256']:
            raise ValueError('Prompt text or embedding changed')
    starts = [x['start_chunk'] for x in request['prompt_schedule']]
    if not starts or starts[0] != 0 or starts != sorted(set(starts)) or starts[-1] >= request['chunks']:
        raise ValueError('Invalid prompt schedule')
    if any(not 0 <= x['prompt_index'] < len(request['prompts']) for x in request['prompt_schedule']):
        raise ValueError('Invalid prompt index')
    return manifest


def main(*, contract=CONTRACT, schema='h3_native_dmd_eval_v1'):
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--verify-checkpoints', action='store_true')
    a = ap.parse_args()
    request = json.loads(Path(a.request).read_text())
    torch.set_num_threads(2)
    if a.preflight_only:
        manifest = validate(request, a.verify_checkpoints, contract=contract, schema=schema)
        identities = {x['weights_spec']['checkpoint']: list((lambda s: (s.st_size, s.st_mtime_ns))(Path(x['weights_spec']['checkpoint']).stat())) for x in request['checkpoints']}
        atomic(Path(a.request).with_name('PREP_READY.json'), dict(request_sha256=sha(a.request),
            checkpoint_files=identities, all_checkpoint_content_hashes_verified=a.verify_checkpoints,
            worker_sha256=sha(__file__), training_source=manifest))
        print('NATIVE_DMD_PREFLIGHT_OK', flush=True)
        return

    import torch.distributed as dist
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    # Import the existing frozen ODE input/position implementation first. It
    # installs the SHA-matched causal base module required by the sink snapshot.
    from h3_stage1_clock_probe import chunk_contract, clocks, verify_frozen, VideoKVBank
    sys.path.insert(0, str(FROZEN))
    from h3_worldattention_causal_sink import WorldAttentionCausalLayout, install_worldattention_causal_attention
    from h3_worldattention_causal import H3CausalContext
    from h3_worldattention_dmd_contract import flow_to_clean, renoise_clean
    from h3_weights_interface_diagnostic import stage1b_segments, stage1b_decode_long, stage1b_context, install_stage1b_processors
    from h3_teacher_rung_diagnostic import prompt_for_model
    from h3_shared_loader import load_session

    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10), device_id=device)
    try:
        if dist.get_world_size() != 8: raise ValueError('Eight cooperative ranks required')
        validate(request, contract=contract, schema=schema)
        verify_frozen()
        prep = json.loads(Path(a.request).with_name('PREP_READY.json').read_text())
        if prep['request_sha256'] != sha(a.request) or not prep['all_checkpoint_content_hashes_verified']:
            raise ValueError('Verified CPU prep required')
        for path, identity in prep['checkpoint_files'].items():
            s = Path(path).stat()
            if [s.st_size, s.st_mtime_ns] != identity: raise ValueError('Checkpoint changed after CPU hash')
        prompts = [torch.load(p['path'], map_location='cpu', weights_only=True) for p in request['prompts']]
        outroot = Path(os.environ['INFER_OUTPUT_DIR'])
        for item in request['checkpoints']:
            began = time.monotonic()
            session = load_session(item['weights_spec'], device, dense=True, native_dmd=contract==CONTRACT,
                stage1b_current_only=contract=='STAGE1B_CURRENT_ONLY_V1',
                load_decoders=rank == 0, force_reshard=True)
            model = session['model']
            from types import SimpleNamespace
            blocks = []
            for wrapped in model.transformer.transformer_blocks:
                block = wrapped
                while hasattr(block, '_checkpoint_wrapped_module'): block = block._checkpoint_wrapped_module
                blocks.append(block)
            if contract == CONTRACT:
                install_worldattention_causal_attention(SimpleNamespace(transformer_blocks=blocks))
            else:
                install_stage1b_processors(model)
            out = outroot / item['id']
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'RESULT.json').exists(): raise FileExistsError('Reuse completed output instead of recomputing')
            schedule = [(0 if i == 0 else 2 + 5*i, 7 + 5*i) for i in range(request['chunks'])]
            frames = 5 + 17*request['chunks']
            vg = torch.zeros(24, schedule[-1][1], 48, 84, dtype=torch.bfloat16)
            ag = torch.zeros(2, 32, round(frames*40/24), dtype=torch.bfloat16)
            bank = VideoKVBank(9, 3)
            generator = torch.Generator(device=device).manual_seed(request['seed'])
            records, archives = [], []
            torch.cuda.reset_peak_memory_stats(device)
            def context(batch, ci, start, end, write=False):
                if contract == 'STAGE1B_CURRENT_ONLY_V1':
                    return stage1b_context(batch,ci,start,end,bank,bank if write else None)
                x = batch['inputs']
                ids = torch.full((len(x['position_ids']),), ci, device=device, dtype=torch.long)
                ids[x['text_indices']] = -1
                vf = torch.full_like(ids, -1)
                vf[x['video_indices']] = torch.arange(start, end, device=device).repeat_interleave(1008)
                return H3CausalContext(WorldAttentionCausalLayout(ids, vf, 12, 3),
                    cached=bank.snapshot(), write_bank=bank if write else None,
                    modulation_segments=stage1b_segments(x['timestep_indices']*3+x['token_tags']))
            def clock(batch, tv, ta):
                x = batch['inputs']
                t = torch.full((len(x['position_ids']),), float(tv), device=device)
                t[x['state_indices']] = 1.; t[x['audio_indices']] = float(ta)
                x['timestep'], x['timestep_indices'] = torch.unique(t, sorted=True, return_inverse=True)
            with torch.inference_mode():
                for ci, (start, end) in enumerate(schedule):
                    pi = next(x['prompt_index'] for x in reversed(request['prompt_schedule']) if ci >= x['start_chunk'])
                    emb = prompts[pi]
                    ae = round((5+17*((end-2)//5))*40/24)
                    batch = chunk_contract(vg, ag, emb['prompt_embeds'], emb['text_token_tags'], generator, device, start, end, ae)
                    prompt_for_model(batch, model)
                    x = batch['inputs']; hashes = [tensor_sha(x['position_ids'])]
                    xv = torch.randn(x['hidden_states'].shape, device=device, generator=generator)
                    xa = torch.randn(x['audio_hidden_states'].shape, device=device, generator=generator)
                    hashes += [tensor_sha(xv), tensor_sha(xa)]
                    tick = time.monotonic()
                    for i in range(4):
                        tv, ta = clocks(1-i/4)
                        x['hidden_states'], x['audio_hidden_states'] = xv, xa
                        clock(batch, tv, ta)
                        pv, pa = model.transformer(**x, attention_mask=context(batch, ci, start, end))
                        cv = flow_to_clean(xv, pv.float(), tv).to(torch.bfloat16)
                        ca = flow_to_clean(xa, pa.float(), ta).to(torch.bfloat16)
                        if i < 3:
                            nv, na = clocks(1-(i+1)/4)
                            ev = torch.randn(cv.shape, device=device, generator=generator)
                            ea = torch.randn(ca.shape, device=device, generator=generator)
                            hashes += [tensor_sha(ev), tensor_sha(ea)]
                            xv, xa = renoise_clean(cv, ev, nv).to(torch.bfloat16), renoise_clean(ca, ea, na).to(torch.bfloat16)
                    if not bool(torch.isfinite(cv).all() and torch.isfinite(ca).all()): raise ValueError('Nonfinite block')
                    x['hidden_states'], x['audio_hidden_states'] = cv, ca
                    clock(batch, 1., 1.)
                    model.transformer(**x, attention_mask=context(batch, ci, start, end, True))
                    peers = [None]*8; dist.all_gather_object(peers, hashes)
                    if any(v != peers[0] for v in peers): raise ValueError('Cooperative ranks differ')
                    torch.cuda.synchronize(device)
                    path = out / f'chunk{ci:03d}.pt'
                    rec = dict(chunk=ci, start_latent=start, end_latent=end, prompt_index=pi,
                        NFE=4, clean_writes=1, old_history_queries_recomputed=0, KV_bytes=bank.bytes,
                        finite=True, input_hashes=hashes, video_sha256=tensor_sha(cv), audio_sha256=tensor_sha(ca),
                        wall_seconds=time.monotonic()-tick)
                    if rank == 0:
                        torch.save(dict(video=cv.cpu(), audio=ca.cpu(), record=rec), path)
                        records.append(rec); archives.append(str(path))
                        atomic(out / 'PROGRESS.json', dict(completed_chunks=ci+1, records=records))
                        print(json.dumps(dict(event='native_dmd_chunk', checkpoint=item['id'], **rec)), flush=True)
                del bank, batch, x, cv, ca, xv, xa, pv, pa
                torch.cuda.empty_cache(); dist.barrier()
                error = [None]
                if rank == 0:
                    try:
                        media = stage1b_decode_long(session, out, archives, frames, device)
                        atomic(out/'RESULT.json', dict(status='complete_visual_review_pending',
                            checkpoint_sha256=item['weights_spec']['checkpoint_sha256'], checkpoint_step=item['step'],
                            attention_contract=contract, source_manifest=prep['training_source'], NFE=4,
                            changed_interface_experiment=item['weights_spec'].get('comparison_track')=='initialization_weights_native_dmd_inference_rules',
                            source_attention_contract=item['weights_spec'].get('initialization_source_attention_contract',contract),
                            sampler='native_H3_stochastic_four_step_x0_renoising', chunks=request['chunks'], FPS=24,
                            frames=frames, seed=request['seed'], request_sha256=sha(a.request), records=records,
                            history_recomputation_allowed=False, old_history_queries_recomputed=0,
                            quality_verified=False, wall_seconds=time.monotonic()-began,
                            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30, **media))
                    except Exception as e: error[0] = repr(e); atomic(out/'EXPORT_ERROR.json', dict(error=error[0]))
                dist.broadcast_object_list(error, src=0)
                if error[0]: raise RuntimeError(error[0])
            del session, model, blocks
            import gc
            gc.collect(); torch.cuda.empty_cache(); dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
