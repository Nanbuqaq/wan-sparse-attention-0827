"""Share repeated preencoded prompt storage in the qualified batch-one T2V path."""
from contextlib import contextmanager
import inspect
import sys
import time
from unittest.mock import patch

import torch


class SharedNativeConditioning:
    """No text/model computation changes; aggregate prefill input is deliberately absent."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.blocks = []
        self.unique_prompts = []
        self.unique_gpu = None
        self.ledger = dict(calls=0, blocks=0, unique_prompts=0, CPU_unique_pack_bytes=0,
            condition_H2D_bytes=0, GPU_owned_storage_bytes=0,
            reference_expanded_GPU_bytes=0, prepare_host_s=0.,
            input_audit_D2H_bytes=0, input_audit_host_s=0., generator_calls_verified=0)

    def encode(self, text_encoder, text_prompts, batch_size):
        if text_encoder is not self.encoder or batch_size != 1 or len(text_prompts) != 1:
            raise ValueError('shared conditioning requires this preencoded encoder and batch one')
        prompts = list(text_prompts[0])
        if not prompts or any(not isinstance(p, str) for p in prompts):
            raise ValueError('nonempty per-block text prompts required')
        resolved = [self.encoder.aliases.get(p, p) for p in prompts]
        unique = list(dict.fromkeys(resolved))
        values = [self.encoder.values[p] for p in unique]
        if any(v.device.type != 'cpu' or v.ndim != 3 or v.shape[0] != 1
               or v.shape != values[0].shape or v.dtype != values[0].dtype for v in values):
            raise ValueError('same-shape preencoded CPU tensors [1,L,D] required')
        began = time.perf_counter()
        packed = torch.cat(values, dim=0)
        self.unique_gpu = packed.to(self.encoder.target_device)
        lookup = {p: i for i, p in enumerate(unique)}
        self.blocks = [{'prompt_embeds': self.unique_gpu[lookup[p]:lookup[p]+1]} for p in resolved]
        self.unique_prompts = unique
        self.ledger.update(calls=self.ledger['calls']+1, blocks=len(prompts), unique_prompts=len(unique),
            CPU_unique_pack_bytes=packed.numel()*packed.element_size(),
            condition_H2D_bytes=packed.numel()*packed.element_size() if self.unique_gpu.is_cuda else 0,
            GPU_owned_storage_bytes=self.unique_gpu.untyped_storage().nbytes() if self.unique_gpu.is_cuda else 0,
            reference_expanded_GPU_bytes=len(prompts)*values[0].numel()*values[0].element_size(),
            prepare_host_s=time.perf_counter()-began)
        # The qualified native T2V loop replaces this aggregate before consuming it.
        # Unsupported prefill must fail instead of receiving a misleading flattened tensor.
        return {}, self.blocks

    def verify_unchanged(self):
        began = time.perf_counter()
        current = self.unique_gpu.cpu()
        if self.unique_gpu.is_cuda:
            self.ledger['input_audit_D2H_bytes'] += current.numel()*current.element_size()
        for index, prompt in enumerate(self.unique_prompts):
            if not torch.equal(current[index:index+1], self.encoder.values[prompt]):
                raise RuntimeError('shared text input changed in generation')
        self.ledger['input_audit_host_s'] += time.perf_counter()-began
        self.ledger['unique_inputs_unchanged'] = True

    def verify_consumption(self, frame_tokens, module, args, kwargs):
        began = time.perf_counter()
        frame = int(kwargs['current_start']) // frame_tokens
        block = frame // 8
        actual = kwargs['conditional_dict']['prompt_embeds']
        expected = self.blocks[block]['prompt_embeds']
        if (actual.data_ptr() != expected.data_ptr() or actual.shape != expected.shape
                or actual.stride() != expected.stride() or actual.dtype != expected.dtype):
            raise RuntimeError('generator did not consume the expected shared block input')
        self.ledger['generator_calls_verified'] += 1
        self.ledger['input_audit_host_s'] += time.perf_counter()-began

    def auxiliary_source_condition(self, *, source_start, current_start, kind):
        """Resolve only an already committed source or the arrived current request."""
        if (kind not in ('past','current') or source_start<0 or source_start%8
                or current_start%8 or source_start+8>current_start
                or current_start//8>=len(self.blocks)):
            raise ValueError('auxiliary source condition violates the causal block boundary')
        block=source_start//8 if kind=='past' else current_start//8
        return self.blocks[block]

    def run_source_auxiliary(self, pipe, latent, condition, caches, cross_caches,
                             *, source_start, current_start, kind):
        """Explicit, audited non-generation call; ordinary hooks are not invoked.

        Calling forward directly prevents an old source reconstruction from
        being counted as a newly generated/decoded chunk. The normal strict
        consumption hook remains unchanged for every ordinary model call.
        """
        expected=self.auxiliary_source_condition(source_start=source_start,current_start=current_start,kind=kind)
        actual=condition['prompt_embeds'];wanted=expected['prompt_embeds']
        if (actual.data_ptr()!=wanted.data_ptr() or actual.shape!=wanted.shape
                or actual.stride()!=wanted.stride() or actual.dtype!=wanted.dtype):
            raise RuntimeError('auxiliary condition differs from its registered causal input')
        if latent.ndim!=5 or latent.shape[:2]!=(1,8) or latent.dtype!=torch.bfloat16:
            raise ValueError('auxiliary reconstruction requires one committed BF16 chunk8')
        if len(caches)!=len(pipe.kv_cache_pos) or len(cross_caches)!=len(pipe.crossattn_cache_pos):
            raise ValueError('auxiliary cache layer count differs')
        native_pointers={v.untyped_storage().data_ptr() for c in pipe.kv_cache_pos+pipe.crossattn_cache_pos
                         for v in c.values() if isinstance(v,torch.Tensor)}
        auxiliary_pointers={v.untyped_storage().data_ptr() for c in caches+cross_caches
                            for v in c.values() if isinstance(v,torch.Tensor)}
        if native_pointers & auxiliary_pointers:
            raise RuntimeError('auxiliary workspace aliases live native state')
        if any(c['k'].shape[1]!=8*pipe.frame_seq_length or c['v'].shape!=c['k'].shape
               or int(c['local_end_index'])!=0
               or int(c['global_end_index'])!=source_start*pipe.frame_seq_length for c in caches):
            raise ValueError('auxiliary workspace must contain exactly8 initially empty frames')
        self.ledger['auxiliary_source_attempts']=self.ledger.get('auxiliary_source_attempts',0)+1
        devices=[latent.device] if latent.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            result=pipe.generator.forward(noisy_image_or_video=latent,conditional_dict=condition,
                timestep=torch.zeros((1,8),device=latent.device,dtype=torch.float32),
                kv_cache=caches,crossattn_cache=cross_caches,
                current_start=source_start*pipe.frame_seq_length,cache_start=source_start*pipe.frame_seq_length)
        self.ledger['auxiliary_source_calls_verified']=self.ledger.get('auxiliary_source_calls_verified',0)+1
        return result

    @contextmanager
    def activate(self, pipe, *, audit_inputs=False):
        if (pipe.guidance_scale != 1 or pipe.num_frame_per_block != 8
                or pipe.independent_first_frame):
            raise ValueError('shared conditioning requires CFG1 and ordinary 8-frame T2V blocks')
        original = pipe.inference
        module = sys.modules[original.__func__.__module__]
        signature = inspect.signature(original)
        original_encoder = module.encode_prompt_blocks

        def dispatch(encoder, prompts, batch_size):
            if encoder is self.encoder:
                return self.encode(encoder, prompts, batch_size)
            return original_encoder(encoder, prompts, batch_size)

        def guarded(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if (bound.arguments['initial_latent'] is not None
                    or bound.arguments['start_frame_index'] not in (None, 0)
                    or bound.arguments['noise'].shape[0] != 1):
                raise ValueError('shared conditioning does not support prefill, continuation, or multi-sample batches')
            with patch.object(module, 'encode_prompt_blocks', dispatch):
                return original(*args, **kwargs)

        hook = None
        if audit_inputs:
            hook = pipe.generator.register_forward_pre_hook(
                lambda m, a, kw: self.verify_consumption(pipe.frame_seq_length, m, a, kw), with_kwargs=True)
        try:
            with patch.object(pipe, 'inference', guarded):
                yield self
        finally:
            if hook is not None:
                hook.remove()

    def audit(self):
        return dict(self.ledger, scope='preencoded CPU text; batch1 CFG1 T2V; unique GPU tensor plus block views',
            aggregate_prefill_supported=False, host_scope_is_not_pure_CPU_arithmetic=True,
            online_routing_or_teacher_inputs_changed=False, text_model_compute_changed=False,
            audit_cost_separately_recorded=True)
