"""Load only each inference rank's H3 parameter slices; retain one session per batch.

The exported full model, released safetensors and adapter states remain separate
formats. No optimizer is loaded, and no dtype conversion is inferred from rank.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from h3_shared_stage1 import CONTRACT as SHARED_STAGE1_CONTRACT
STAGE1_ODE_V2_CONTRACT = 'chunk_causal_local12_readonly_text_no_audio_history_full_sequence_ODE_v2'

from h3_legacy_fixed_kv import TRACK as LEGACY_KV_TRACK, require_migration_spec, commit_projection
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


def canonical_name(name):
    for prefix in ('._checkpoint_wrapped_module', '._offload_wrapped_module'):
        name = name.replace(prefix, '')
    return name


class WeightSource:
    def __init__(self, spec):
        self.spec = spec
        self.kind = spec['kind']
        self.stack = ExitStack()
        self.handles = {}
        self.full = None
        self.adapters = {}
        self.training_metadata = None
        self.step = 0
        self.legacy_fixed_kv_migration = spec.get('attention_contract') == SHARED_STAGE1_CONTRACT and spec.get('comparison_track') == LEGACY_KV_TRACK
        if self.legacy_fixed_kv_migration:
            require_migration_spec(spec)
        if self.kind == 'full':
            self.full = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
            if self.full.get('schema') != 'h3_full_dit_inference_v1':
                raise ValueError('Expected a model-only full-DiT export')
            expected = (STAGE1_ODE_V2_CONTRACT if spec.get('attention_contract') == STAGE1_ODE_V2_CONTRACT
                        else SHARED_STAGE1_CONTRACT if spec.get('attention_contract') == SHARED_STAGE1_CONTRACT
                        else 'text_current_full_history_prefix_recomputed_v1')
            actual = self.full.get('attention_contract')
            if actual != expected:
                if not (expected == SHARED_STAGE1_CONTRACT and spec.get('comparison_track') == 'legacy_weights_new_fixed_kv_interface'
                        and actual == 'text_current_full_history_prefix_recomputed_v1'):
                    raise ValueError('Checkpoint and requested inference attention contracts differ')
            if actual == SHARED_STAGE1_CONTRACT:
                if self.legacy_fixed_kv_migration:
                    raise ValueError('New shared-KV checkpoints must not use the legacy projection policy')
                arm = self.full.get('metadata', {}).get('args', {}).get('ablation_arm')
                if arm != spec.get('strategy'):
                    raise ValueError('Shared stage1 checkpoint strategy is missing or mismatched')
            self.step = self.full['step']
        else:
            root = Path(spec['base_source']) / 'transformer'
            index = json.loads((root / 'diffusion_pytorch_model.safetensors.index.json').read_text())
            self.index, self.root = index['weight_map'], root
            if self.kind == 'adapter':
                saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
                self.adapters = {canonical_name(k): v for k, v in saved['adapters'].items()}
                self.step = saved['step']
                self.training_metadata = {k:saved.get(k) for k in ['schema','training_kind','history_policy','args','index_sha256','generated_history_training']}
            elif self.kind != 'base':
                raise ValueError(self.kind)

    def _base_key(self, name):
        return name.removeprefix('transformer.').replace('.base.', '.')

    def dtype(self, name):
        name = canonical_name(name)
        if name.startswith('projection.') or name.endswith(('.lora_a', '.lora_b')):
            return torch.float32
        key = self._base_key(name)
        if self.kind == 'full':
            return self.full['transformer'][key].dtype
        if self.spec.get('full_native_precision') and key.startswith('norm_out.'):
            return torch.float32
        # Match the released Diffusers loader's explicit keep-in-FP32 list.
        return torch.float32 if any(x in key for x in ('proj_in', 'time_embedder', 'proj_out')) else torch.bfloat16

    def tensor(self, name, start=None, stop=None):
        name = canonical_name(name)
        if name in self.adapters:
            value = self.adapters[name]
        elif name.startswith('projection.'):
            key = name.removeprefix('projection.')
            if self.kind == 'full':
                value = self.full['history_projection'][key]
            else:
                return self.tensor('transformer.proj_in.' + key, start, stop).float()
        elif name.endswith(('.lora_a', '.lora_b')):
            # A zero-B untrained branch is algebraically independent of A.
            # Initialized zero A avoids uninitialized/NaN scratch parameters.
            return None
        elif self.kind == 'full':
            value = self.full['transformer'][self._base_key(name)]
        else:
            from safetensors import safe_open
            key = self._base_key(name)
            file = self.index[key]
            if file not in self.handles:
                self.handles[file] = self.stack.enter_context(safe_open(str(self.root / file), framework='pt', device='cpu'))
            view = self.handles[file].get_slice(key)
            return view[:] if start is None else view[start:stop]
        return value if start is None else value[start:stop]

    def close(self):
        self.stack.close()
        self.full = None
        self.adapters.clear()


def build_meta_model(source, dense=False):
    from diffusers import MiniMaxH3Transformer3DModel
    from h3_fixed_chunk_contract import FixedTrainingModel
    from h3_spatial_history_flow import SpatialHistoryProjection
    from train_h3_v2_local_history_ddp import install_lora

    class MetaProjection(SpatialHistoryProjection):
        def __init__(self, native):
            torch.nn.Linear.__init__(self, native.in_features, native.out_features,
                                     bias=native.bias is not None, device='meta', dtype=torch.float32)

    shared = source.spec.get('attention_contract') == SHARED_STAGE1_CONTRACT
    legacy_migration = source.legacy_fixed_kv_migration
    class BatchModel(FixedTrainingModel):
        def __init__(self, transformer):
            torch.nn.Module.__init__(self)
            self.transformer = transformer
            self.projection = MetaProjection(transformer.proj_in)

        @property
        def config(self):
            return self.transformer.config

        @property
        def transformer_blocks(self):
            return self.transformer.transformer_blocks

        def forward(self, *args, **kwargs):
            if dense:
                return self.transformer(*args, **kwargs)
            if shared:
                from h3_fixed_kv_stage1 import forward
                batch = args[0]
                if legacy_migration:
                    with commit_projection(self.transformer, self.projection, batch['inputs'], batch['context']):
                        return forward(self.transformer, batch['inputs'], batch['context'])
                return forward(self.transformer, batch['inputs'], batch['context'])
            return super().forward(*args, **kwargs)

    config = MiniMaxH3Transformer3DModel.load_config(source.spec['base_source'], subfolder='transformer')
    with torch.device('meta'):
        transformer = MiniMaxH3Transformer3DModel.from_config(config)
        if (source.kind in ('base', 'adapter') and not dense and
                ((not shared and source.spec.get('zero_lora_interface',True)) or
                 (legacy_migration and source.kind == 'adapter'))):
            install_lora(transformer, 8, 8)
        model = BatchModel(transformer)
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to(source.dtype(name))
    model.requires_grad_(False)
    return model


def load_session(spec, device, *, dense=False, load_decoders=True, diagnostic_native_rules=False, stage1_ode_v2=False):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    from h3_fsdp2_speed import shard_speed_model, apply_variant
    from h3_prompt_history_attention import install_prompt_history_attention
    from h3_fixed_stream_decode import FixedChunkVideoDecoder
    started = time.monotonic()
    rank, world = dist.get_rank(), dist.get_world_size()
    shared = spec.get('attention_contract') == SHARED_STAGE1_CONTRACT
    if spec.get('attention_contract') == STAGE1_ODE_V2_CONTRACT:
        if not (stage1_ode_v2 and dense and spec.get('kind') == 'full' and spec.get('diagnostic_only') is True):
            raise ValueError('Stage1 ODE-v2 requires an explicit matched causal diagnostic caller')
        if diagnostic_native_rules:
            raise ValueError('Stage1 causal checkpoint must not silently use native dense rules')
    elif stage1_ode_v2:
        raise ValueError('Stage1 ODE-v2 mode requires the exact checkpoint contract')
    if diagnostic_native_rules:
        if not dense or spec.get('kind') not in ('base', 'full'):
            raise ValueError('Native-rule diagnosis requires a dense base/full model')
        # Keep WeightSource's checkpoint-schema/arm validation. Only the
        # execution rule changes, explicitly, in an isolated diagnosis.
        # Production callers retain the existing contract checks unchanged.
        shared = False
    shared_manifest = None
    if shared:
        if dense or spec['kind'] not in ['base', 'full', 'adapter'] or spec.get('strategy') not in ['A', 'B', 'C']:
            raise ValueError('Use original native H3 or matched full DiT for the shared A/B/C interface')
        if spec['kind'] == 'adapter':
            require_migration_spec(spec)
        from h3_shared_stage1 import shared_modules
        shared_code, shared_manifest = shared_modules(spec.get('shared_stage1_root'))
    source = WeightSource(spec)
    model = build_meta_model(source, dense=dense)
    if source.adapters:
        unused = set(source.adapters) - {canonical_name(n) for n, _ in model.named_parameters()}
        if unused:
            raise ValueError('Checkpoint adapter parameters are not represented in the inference model: '+str(sorted(unused)))
    transformer = model.transformer
    if shared:
        processors = shared_code['h3_fixed_kv_stage1'].install(transformer)
        shared_code['h3_stable_modulation'].install_stable_modulation(transformer)
        if spec.get('head_sharded_same_video') is not True:
            raise ValueError('The shared inference adapter requires an explicit same-video head-sharded group')
        from h3_shared_kv_heads import install_head_shards
        head_handles = install_head_shards(transformer.transformer_blocks)
    else:
        processors = [] if dense else install_prompt_history_attention(transformer, 8192)
    mesh = init_device_mesh('cuda', (world,))
    reshard = torch.cuda.get_device_properties(device).total_memory < 100 * 2**30
    # Full exports preserve the native FP32 norm_out, unlike old adapter runs.
    fully_shard(transformer.norm_out, mesh=mesh, mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False),
                reshard_after_forward=reshard)
    model, blocks, wrappers = shard_speed_model(model, mesh, reshard_after_forward=reshard)
    apply_variant('no_inner_8k', blocks, wrappers, processors)
    if dense:
        for block in blocks:
            block.ff_chunk_size, block.ff_chunk_checkpoint = 8192, False
    model.to_empty(device=device)
    loaded, local_elements = 0, 0
    fingerprints = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            target = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            start = stop = None
            if isinstance(parameter, DTensor):
                if len(parameter.placements) != 1:
                    raise ValueError('Only a one-dimensional inference mesh is supported')
                placement = parameter.placements[0]
                if isinstance(placement, Shard):
                    if placement.dim != 0:
                        raise ValueError('Unexpected FSDP shard dimension')
                    size = (parameter.shape[0] + world - 1) // world
                    start = min(rank * size, parameter.shape[0])
                    stop = min(start + size, parameter.shape[0])
                elif not isinstance(placement, Replicate):
                    raise ValueError('Unsupported parameter placement')
            value = source.tensor(name, start, stop)
            if value is None:
                target.zero_()
            else:
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(f'Shard shape mismatch {name}: {value.shape} != {target.shape}')
                target.copy_(value.to(device=device, dtype=target.dtype))
            if target.numel():
                sample = target.detach().reshape(-1)[::max(1, target.numel() // 16)][:16].float()
                if not torch.isfinite(sample).all():
                    raise ValueError('Nonfinite loaded parameter sample: ' + name)
                fingerprints.append((canonical_name(name), sample.cpu().tolist()))
            loaded += 1
            local_elements += target.numel()
            if loaded % 64 == 0:
                print(json.dumps(dict(event='parameter_slices_loaded', rank=rank, tensors=loaded,
                                      local_elements=local_elements, seconds=time.monotonic()-started)), flush=True)
    n = transformer.config.rope_freq_dim
    transformer.rope.inv_freq = 1. / (transformer.config.rope_theta ** (
        torch.arange(0, 2*n, 2, device=device, dtype=torch.float32) / (2*n)))
    if any(value.is_meta for value in model.buffers()):
        raise ValueError('An uninitialized meta buffer remains')
    step = source.step
    training_metadata = source.full['metadata'] if source.kind == 'full' else source.training_metadata
    source.close()
    model.eval()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(spec['base_source'], subfolder='vae').to(device).eval() if load_decoders else None
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(spec['base_source'], subfolder='audio_vae').to(device).eval() if load_decoders else None
    if reshard and load_decoders:
        vae.cpu(); audio_vae.cpu(); torch.cuda.empty_cache()
    return dict(model=model, processors=processors, vae=vae, audio_vae=audio_vae,
                decoder=FixedChunkVideoDecoder(vae,device) if load_decoders else None, reshard=reshard,
                load_seconds=time.monotonic()-started, checkpoint_step=step,
                checkpoint_sha256=spec.get('checkpoint_sha256'), weights_kind=spec['kind'],
                training_metadata=training_metadata, tensors_loaded=loaded,
                local_elements=local_elements, parameter_samples=fingerprints,
                shared_stage1_manifest=shared_manifest,
                attention_contract=spec.get('attention_contract','text_current_full_history_prefix_recomputed_v1'),
                diagnostic_native_rules=diagnostic_native_rules)
