#!/usr/bin/env python3
"""Native fresh T2V/I2V suffix control, explicitly separate from stream KV recall.

The spec contains an arrived request and one causally selected past RGB image.
No initial latent from a temporal video is reused as a single-image latent.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_longlive2_native_reference import CachedNativeTextEncoder, SOURCE_SHA
from adapters.longlive_sparse.history_cache import tensor_sha256


def read_spec(path):
    spec = json.loads(path.read_text())
    if spec['schema'] != 'causal_visual_restart_v1':
        raise ValueError('unregistered visual restart spec')
    if not 0 <= spec['source_pixel_index'] <= 4 * spec['source_committed_latents'] - 4:
        raise ValueError('source image is beyond its committed prefix')
    if spec['source_committed_latents'] > spec['request_arrival_latent']:
        raise ValueError('future source is prohibited')
    if not spec['prompt'] or spec['prompt'].startswith('The scene transitions. '):
        raise ValueError('fresh clip needs the arrived user request without native cut marker')
    image = path.parent / spec['image']
    if hashlib.sha256(image.read_bytes()).hexdigest() != spec['image_sha256']:
        raise ValueError('source image changed')
    return spec, image


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'third_party/LongLive2')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('t2v', 'i2v'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gate', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    args.assets=args.assets.resolve(); args.source=args.source.resolve()
    args.spec=args.spec.resolve(); args.output=args.output.resolve()
    spec, image_path = read_spec(args.spec)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', mode=args.mode, seed=args.seed, spec=spec,
                  scope='fresh 32-latent suffix; no live-stream cache or archive restored',
                  source_spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                  code_sha=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(0), pixel_gpu=torch.cuda.get_device_name(1),
                  physical_GPUs=os.environ.get('WAN_SPARSE_PHYSICAL_GPUS'), gate=args.gate,
                  fixed_latent_frames=1 if args.mode == 'i2v' else 0,
                  total_latent_frames=32, pixel_frames=125,
                  primary_quality_pixels=[29, 124], first_chunk_excluded_from_primary_quality=True,
                  full_stream_delivery_claim=False, shared_T2V_conditioning_guard_changed=False)
    pipe = video_pipeline = inplace = hook = None
    started = time.perf_counter()
    try:
        if torch.cuda.device_count() != 2 or not report['physical_GPUs']:
            raise RuntimeError('two coordinated physical GPUs required')
        source_sha = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
        if source_sha != SOURCE_SHA:
            raise ValueError('native source lock changed')
        if json.loads((args.assets/'assets_manifest.json').read_text())['status'] != 'pass':
            raise ValueError('verified existing assets required')
        for name in ('LLV2_USE_FA3', 'LLV2_USE_FA4', 'LLV2_USE_TE_ATTN'):
            if os.environ.get(name, '0') != '0':
                raise ValueError('native FA2 required')
        sys.path.insert(0, str(args.source)); os.chdir(args.assets)
        from omegaconf import OmegaConf
        from pipeline import CausalDiffusionInferencePipeline
        from utils.config import normalize_config
        from utils.wan_5b_wrapper import CausalWanModel
        from utils.inference_utils import load_generator_checkpoint
        from wan_5b.modules.vae2_2 import unpatchify
        import wan_5b.modules.causal_model as native
        import wan_5b.modules.attention as native_attention
        if not native_attention.FLASH_ATTN_2_AVAILABLE:
            raise RuntimeError('no fallback permitted')
        from adapters.longlive_sparse.strict_checkpoint_init import StrictCheckpointParameterInit
        from adapters.longlive_sparse.native_capacity import install_positive_only_allocator
        from adapters.longlive_sparse.native_inplace_cache import NativeInplaceCache
        from adapters.longlive_sparse.native_inplace_gelu import NativeInplaceGelu
        from adapters.longlive_sparse.native_kernel_recipe import fix_native_adaln_for_fresh_run
        from adapters.longlive_sparse.native_video_pipeline import NativeVideoPipeline
        from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink
        fix_native_adaln_for_fresh_run(16, 1)
        raw = OmegaConf.load(args.source/'configs/inference.yaml')
        del raw.adapter
        raw.checkpoints.lora_ckpt = None
        raw.checkpoints.generator_ckpt = str(args.assets/'checkpoints/model_bf16.pt')
        raw.inference.streaming_vae = False; raw.inference.async_vae = False; raw.inference.vae_device = None
        raw.inference.independent_first_frame = args.mode == 'i2v'
        raw.model_kwargs.local_attn_size = 32; raw.inference.local_attn_size = 32
        h, w = (16, 32) if args.gate else (44, 80)
        raw.data.image_or_video_shape = [1, 32, 48, h, w]
        OmegaConf.save(raw, args.output/'config.yaml')
        def architecture(path, **kw):
            return CausalWanModel.from_config(json.loads((Path(path)/'config.json').read_text()), **kw)
        with StrictCheckpointParameterInit(enabled=True):
            with patch.object(CausalWanModel, 'from_pretrained', side_effect=architecture):
                pipe = CausalDiffusionInferencePipeline(normalize_config(raw), device=torch.device('cuda'))
            pipe.text_encoder.to(dtype=torch.bfloat16)
            loaded = load_generator_checkpoint(pipe.generator, str(args.assets/'checkpoints/model_bf16.pt'), strict=True)
            if loaded.missing_keys or loaded.unexpected_keys:
                raise RuntimeError('strict generator load failed')
        report['load_s'] = time.perf_counter()-started
        began = time.perf_counter()
        encoded = pipe.text_encoder(text_prompts=[spec['prompt']])['prompt_embeds'].detach().cpu()
        torch.cuda.synchronize(0)
        report['T5_s'] = time.perf_counter()-began
        pipe.text_encoder.to('cpu'); pipe.text_encoder = CachedNativeTextEncoder({spec['prompt']: encoded}, torch.device('cuda:0'))
        gc.collect(); torch.cuda.empty_cache()
        pipe.generator.to(device='cuda:0', dtype=torch.bfloat16).eval().requires_grad_(False)
        gelu = NativeInplaceGelu(pipe._dit_model)
        pipe.vae.to(device='cuda:1', dtype=torch.bfloat16)
        torch.cuda.synchronize(1)
        install_positive_only_allocator(pipe)
        inplace = NativeInplaceCache(pipe); inplace.attach()
        torch.cuda.reset_peak_memory_stats(0); torch.cuda.reset_peak_memory_stats(1)
        preparation_start = time.perf_counter()
        initial = None
        if args.mode == 'i2v':
            with Image.open(image_path) as im:
                im = im.convert('RGB')
                report['stored_image_dimensions'] = list(im.size)
                if im.size != (w*16, h*16):
                    if not args.gate:
                        raise ValueError('full source image must have native dimensions')
                    im = im.resize((w*16, h*16), Image.Resampling.LANCZOS)
                pixels = torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1)[None, :, None]
            pixels = pixels.float().div_(255.).sub_(.5).div_(.5).to(device='cuda:1', dtype=torch.bfloat16)
            began = time.perf_counter()
            with torch.cuda.device(1):
                initial_pixel_device = pipe.vae.encode_to_latent(pixels).to(dtype=torch.bfloat16)
                pipe.vae.model.clear_cache()
            torch.cuda.synchronize(1)
            report['native_image_encode_s'] = time.perf_counter()-began
            began = time.perf_counter()
            initial = initial_pixel_device.to('cuda:0')
            torch.cuda.synchronize(0)
            report['initial_latent_transfer_s'] = time.perf_counter()-began
            if tuple(initial.shape) != (1, 1, 48, h, w):
                raise RuntimeError('native image encoder shape differs')
            report['initial_latent_sha256'] = tensor_sha256(initial)
            report['initial_latent_bytes'] = initial.numel()*initial.element_size()
            torch.save(initial.cpu(), args.output/'conditioning_latent.pt')
            del pixels, initial_pixel_device
        report['source_prepare_s'] = time.perf_counter()-preparation_start
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        noise = torch.randn(1, 32, 48, h, w, device='cuda:0', dtype=torch.bfloat16)
        report['noise_sha256'] = tensor_sha256(noise)
        report['generator_inputs'] = []
        def verify_input(module, values, kw):
            frame = int(kw['current_start'])//pipe.frame_seq_length
            row = dict(frame=frame, frames=kw['noisy_image_or_video'].shape[1])
            if initial is not None and frame == 0:
                if not torch.equal(kw['noisy_image_or_video'][:, :1], initial) or not bool((kw['timestep'][:, :1] == 0).all()):
                    raise RuntimeError('native clamp or per-frame timestep failed')
                row['initial_latent_and_timestep_exact'] = True
            report['generator_inputs'].append(row)
        hook = pipe.generator.register_forward_pre_hook(verify_input, with_kwargs=True)
        attention_rows = []
        original_attention = native.attention
        def observed_attention(q, k, v, *a, **kw):
            attention_rows.append(dict(Q=q.shape[1], K=k.shape[1], heads=q.shape[2],
                                       pairs=q.shape[0]*q.shape[1]*k.shape[1]*q.shape[2]))
            return original_attention(q, k, v, *a, **kw)
        generation_started = time.perf_counter()
        sink = IncrementalVideoSink(args.output/'video.mp4', expected_frames=125, started=generation_started, fps=24)
        video_pipeline = NativeVideoPipeline(pipe.vae, unpatchify, sink, source_device='cuda:0', target_device='cuda:1',
            latent_shape=(1,32,48,h,w), started=generation_started, slots=2,
            pinned_budget=128*1024**2, encode_mode='thread', pixel_slots=2)
        video_pipeline.attach(pipe)
        (args.output/'progress.json').write_text(json.dumps(report, indent=2))
        with patch.object(native, 'attention', observed_attention):
            latent = pipe.inference(noise=noise, text_prompts=[[spec['prompt']]*4], initial_latent=initial, return_latents=True)
        torch.cuda.synchronize(0)
        report['generation_s'] = time.perf_counter()-generation_started
        video_pipeline.detach()
        report['pixels'], report['video_pipeline'] = video_pipeline.finish(generation_finished_s=report['generation_s'])
        report['suffix_delivery_with_image_prepare_s'] = report['source_prepare_s']+report['video_pipeline']['complete_s']
        report['observed_self_attention_calls'] = len(attention_rows)
        report['actual_self_attention_pairs_including_clean'] = sum(r['pairs'] for r in attention_rows)
        (args.output/'attention_shapes.json').write_text(json.dumps(attention_rows))
        if len(attention_rows) != 4*5*30 or len(report['generator_inputs']) != 20:
            raise RuntimeError('incomplete real native attention/clean coverage')
        if initial is not None and not torch.equal(latent[:, :1], initial):
            raise RuntimeError('returned initial latent changed')
        if not bool(torch.isfinite(latent).all()):
            raise RuntimeError('nonfinite latent')
        report['latent_sha256'] = tensor_sha256(latent)
        if report['video_pipeline']['streamed_latent_sha256'] != report['latent_sha256']:
            raise RuntimeError('streamed clean latent differs from final output')
        torch.save(latent.cpu(), args.output/'latents.pt')
        video_pipeline.write_trace(args.output/'pipeline_host_trace.json')
        report['GPU0_peak_bytes'] = torch.cuda.max_memory_allocated(0)
        report['GPU1_peak_bytes'] = torch.cuda.max_memory_allocated(1)
        report['native_cache_bytes'] = sum(c[k].numel()*c[k].element_size() for c in pipe.kv_cache_pos for k in ('k','v'))
        report['inplace_cache'] = inplace.audit(); report['inplace_gelu'] = gelu.audit()
        report['wall_including_loading_s'] = time.perf_counter()-started
        report['status'] = 'pass'
    except BaseException:
        report.update(status='fail', traceback=traceback.format_exc())
        if video_pipeline is not None:
            video_pipeline.abort()
        raise
    finally:
        if hook is not None: hook.remove()
        if inplace is not None: inplace.detach()
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: report.get(k) for k in ('status','mode','gate','generation_s','suffix_delivery_with_image_prepare_s')}), flush=True)


if __name__ == '__main__':
    main()
