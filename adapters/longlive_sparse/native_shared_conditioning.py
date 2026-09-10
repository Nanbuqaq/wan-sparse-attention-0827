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
