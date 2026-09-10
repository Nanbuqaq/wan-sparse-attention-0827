"""Opt-in 5B resident-history research bridge, independent of the RAG runtime.

Only committed per-frame Block64 summaries enter selection. The original native
Q/K/V, projections, cache writes and absolute RoPE remain authoritative. This
first bridge selects one shared token set across heads; it is not an offload or
per-query sparse backend. Native mode never installs it.
"""
import ast
from dataclasses import dataclass
import hashlib
import inspect
import math
import textwrap
import time

import torch


@dataclass(frozen=True)
class NativeResidentConfig:
    policy: str = 'identity'
    fraction: float = .25
    block_tokens: int = 64
    query_samples: int = 32
    reuse: str = 'none'
    summary_backend: str = 'scalar'

    def __post_init__(self):
        if self.policy not in ('identity', 'mass_value', 'contrast_value', 'recent'):
            raise ValueError('unknown resident-history policy')
        if not 0 < self.fraction <= 1 or self.block_tokens != 64 or self.query_samples < 1:
            raise ValueError('invalid registered budget/geometry')
        if self.reuse not in ('none', 'denoise_first'):
            raise ValueError('unknown route reuse policy')
        if self.summary_backend not in ('scalar','vectorized'):
            raise ValueError('unknown summary preparation backend')


def updated_frame_slots(previous, info, current_start, frame_tokens):
    """Mirror only the native ownership update, using actual returned metadata."""
    keys = ('local_start_index', 'local_end_index')
    if any(info[k] % frame_tokens for k in keys) or current_start % frame_tokens:
        raise ValueError('native updates must contain whole frames')
    slots = list(previous)
    if info['action'] == 'roll_and_insert':
        if any(info[k] % frame_tokens for k in ('sink_tokens', 'num_evicted_tokens', 'num_rolled_tokens')):
            raise ValueError('unaligned native roll')
        sink, evict, count = (info[k] // frame_tokens for k in
                              ('sink_tokens', 'num_evicted_tokens', 'num_rolled_tokens'))
        slots[sink:sink+count] = slots[sink+evict:sink+evict+count]
    elif info['action'] != 'direct_insert':
        raise ValueError('unregistered native update')
    begin, end = (info[k] // frame_tokens for k in keys)
    slots[begin:end] = range(current_start // frame_tokens, current_start // frame_tokens + end-begin)
    if end > len(previous) or begin < 0:
        raise ValueError('native ownership outside physical capacity')
    return slots


def window_slot_indices(*, end, start, effective_sink, pinned_start, pinned_len,
                        prepend_sink, prepend_pinned, max_tokens, frame_tokens):
    if any(x % frame_tokens for x in (end, start, effective_sink, pinned_len, max_tokens)):
        raise ValueError('unaligned native window')
    spans = []
    if prepend_sink and prepend_pinned:
        spans = [(0, effective_sink), (pinned_start, pinned_start+pinned_len),
                 (max(effective_sink, end-(max_tokens-effective_sink-pinned_len)), end)]
    elif prepend_sink:
        spans = [(0, effective_sink), (max(effective_sink, end-(max_tokens-effective_sink)), end)]
    elif prepend_pinned:
        spans = [(pinned_start, pinned_start+pinned_len), (max(0, end-(max_tokens-pinned_len)), end)]
    else:
        spans = [(start, end)]
    indices = [i for a,b in spans for i in range(a//frame_tokens, b//frame_tokens)]
    if len(set(indices)) != len(indices):
        raise ValueError('native window duplicates physical slots')
    return indices


def summarize_frame(k, v, block_tokens=64):
    """Executed on newly clean-committed KV only; includes true frame tails."""
    sizes = list(range(0, k.shape[0], block_tokens))
    key = torch.stack([k[a:a+block_tokens].float().mean(0) for a in sizes])
    value = torch.stack([v[a:a+block_tokens].float().mean(0) for a in sizes])
    count = torch.tensor([min(block_tokens, k.shape[0]-a) for a in sizes], device=k.device)
    return key, value, count


def contrast_scores(q, key_mean, value_mean, counts, policy, samples=32):
    """Proxy softmax is over eligible historical groups ONLY, not full attention.

    q [Q,H,D], summaries [G,H,D]. Average AFTER nonlinear scoring, first over
    sampled real queries then heads. Never consumes raw historical candidate KV.
    """
    if policy not in ('mass_value', 'contrast_value'):
        raise ValueError('not a scoring policy')
    sites = torch.linspace(0, q.shape[0]-1, min(samples, q.shape[0]), device=q.device).round().long()
    query = q.index_select(0, sites).float()
    logits = torch.einsum('qhd,ghd->hqg', query, key_mean.float()) / math.sqrt(q.shape[-1])
    mass = (logits + counts.float().log()[None,None,:]).softmax(-1)
    values = value_mean.float().permute(1,0,2)
    if policy == 'mass_value':
        score = mass * values.norm(dim=-1)[:,None,:]
    else:
        mixture = torch.einsum('hqg,hgd->hqd', mass, values)
        difference = (values[:,None,:,:] - mixture[:,:,None,:]).norm(dim=-1)
        score = mass * difference
    return score.mean((0,1))


def choose_whole_blocks(scores, counts, fraction):
    budget = math.floor(sum(counts) * fraction)
    selected, used = [], 0
    # Stable ties use chronological (frame, within-frame block) order.
    order = sorted(range(len(counts)), key=lambda i: (-scores[i], i))
    for i in order:
        if used + counts[i] <= budget:
            selected.append(i)
            used += counts[i]
    return sorted(selected), used, budget


class NativeResidentHistory:
    def __init__(self, pipe, config):
        if pipe.use_relative_rope or pipe.guidance_scale != 1 or pipe.quantize_kv:
            raise ValueError('bridge requires absolute-RoPE BF16 CFG1')
        if pipe.num_frame_per_block != 8 or pipe.sampling_steps != 4:
            raise ValueError('bridge requires native block8/four denoising steps')
        if pipe.generator._compiled_model_call is not None:
            raise ValueError('compiled native model needs a separate integration gate')
        self.pipe, self.config = pipe, config
        self.frame_tokens = pipe.frame_seq_length
        self.layers = list(pipe._dit_model.blocks)
        self.slots = [[-1]*pipe.local_attn_size for _ in self.layers]
        self.summaries = [{} for _ in self.layers]
        self.pending = {}
        self.routes = {}
        self.rows = []
        self.handles = []
        self.clean = False
        self.active_start = None
        self.calls = 0
        self.summary_build_host_s = 0.
        self.summary_peak_bytes = 0
        self._native = self._forward = None

    def before(self, owner, values, kwargs):
        if self.pending:
            raise RuntimeError('uncommitted previous call')
        self.active_start = int(kwargs['current_start'])
        # Read one timestep for the whole model; charge this synchronization.
        self.clean = bool((kwargs['timestep'] == 0).all())
        self.calls += 1

    def after(self, owner, values, kwargs, result):
        for layer, (slots, additions) in self.pending.items():
            self.slots[layer] = slots
            bank = self.summaries[layer]
            bank.update(additions)
            valid = set(slots)
            for frame in list(bank):
                if frame not in valid:
                    del bank[frame]
        self.pending.clear()
        size = sum(t.numel()*t.element_size() for bank in self.summaries for value in bank.values() for t in value)
        self.summary_peak_bytes = max(size, self.summary_peak_bytes)
        self.active_start = None

    def dispatch(self, layer, original, q, k, v, *, info, current_start, window_start,
                 effective_sink, pinned_start, pinned_len, prepend_sink, prepend_pinned,
                 max_tokens, cache_end, global_sink_tokens):
        if current_start != self.active_start or q.shape[0] != 1 or k.dtype != torch.bfloat16:
            raise ValueError('unregistered native dispatch identity')
        if info is None or k.shape != v.shape:
            raise ValueError('missing native update / KV shape')
        started = time.perf_counter()
        slots = updated_frame_slots(self.slots[layer], info, current_start, self.frame_tokens)
        physical = window_slot_indices(end=cache_end, start=window_start, effective_sink=effective_sink,
            pinned_start=pinned_start, pinned_len=pinned_len, prepend_sink=prepend_sink,
            prepend_pinned=prepend_pinned, max_tokens=max_tokens, frame_tokens=self.frame_tokens)
        frame_ids = tuple(slots[i] for i in physical)
        if len(frame_ids)*self.frame_tokens != k.shape[1] or any(f < 0 for f in frame_ids):
            raise RuntimeError('native actual KV length and ownership disagree')
        current_frame = current_start // self.frame_tokens
        protected, eligible = [], []
        for window_frame, (physical_frame, frame) in enumerate(zip(physical, frame_ids)):
            pos = physical_frame*self.frame_tokens
            actual_pinned_start = pinned_start-info.get('pinned_shift', 0)
            pinned = pinned_len > 0 and actual_pinned_start <= pos < actual_pinned_start+pinned_len
            keep = frame >= current_frame or pos < max(effective_sink, global_sink_tokens) or pinned
            if keep:
                protected.extend(range(window_frame*self.frame_tokens, (window_frame+1)*self.frame_tokens))
            else:
                eligible.append((window_frame, frame))
        candidate_tokens = len(eligible)*self.frame_tokens
        route_key = (current_start, frame_ids, tuple(protected), self.config.policy, self.config.fraction)
        reuse = self.config.reuse == 'denoise_first' and not self.clean and self.routes.get(layer, (None,))[0] == route_key
        selection_host_s = 0.
        if self.config.policy == 'identity' or not eligible or self.config.fraction == 1:
            keep_indices = None
            selected_tokens = candidate_tokens
        elif reuse:
            _, keep_indices, selected_tokens = self.routes[layer]
        else:
            select_started = time.perf_counter()
            key_means, value_means, sizes, block_tokens = [], [], [], []
            bank = self.summaries[layer]
            for window_frame, frame in eligible:
                if frame not in bank:
                    raise RuntimeError('eligible history has no previously committed summary')
                km, vm, count = bank[frame]
                key_means.append(km); value_means.append(vm); sizes.append(count)
                for start in range(0, self.frame_tokens, self.config.block_tokens):
                    block_tokens.append(list(range(window_frame*self.frame_tokens+start,
                        window_frame*self.frame_tokens+min(start+self.config.block_tokens, self.frame_tokens))))
            km, vm, count = torch.cat(key_means), torch.cat(value_means), torch.cat(sizes)
            if self.config.policy == 'recent':
                scores = list(range(len(block_tokens)))
            else:
                scores = contrast_scores(q[0], km, vm, count, self.config.policy, self.config.query_samples).cpu().tolist()
            selected, selected_tokens, _ = choose_whole_blocks(scores, [len(x) for x in block_tokens], self.config.fraction)
            tokens = sorted(protected + [t for i in selected for t in block_tokens[i]])
            keep_indices = torch.tensor(tokens, device=k.device, dtype=torch.long)
            if not self.clean:
                self.routes[layer] = (route_key, keep_indices, selected_tokens)
            selection_host_s = time.perf_counter()-select_started
        prepare_host_s = time.perf_counter()-started
        with torch.cuda.nvtx.range('native_history/materialize_and_attention'):
            if keep_indices is None:
                output = original(q, k, v)
                actual_tokens = k.shape[1]
            else:
                selected_k = k.index_select(1, keep_indices)
                selected_v = v.index_select(1, keep_indices)
                output = original(q, selected_k, selected_v)
                actual_tokens = len(keep_indices)
        additions = {}
        if self.clean:
            committed_started = time.perf_counter()
            # Current clean KV only. No selected/teacher feedback in this summary.
            new_k, new_v = info['new_k'][0], info['new_v'][0]
            for offset in range(0, new_k.shape[0], self.frame_tokens):
                frame = current_frame+offset//self.frame_tokens
                summary_fn=summarize_frame
                if self.config.summary_backend=='vectorized':
                    from .native_summary_vectorized import summarize_frame_vectorized
                    summary_fn=summarize_frame_vectorized
                additions[frame] = summary_fn(new_k[offset:offset+self.frame_tokens],
                    new_v[offset:offset+self.frame_tokens], self.config.block_tokens)
            self.summary_build_host_s += time.perf_counter()-committed_started
        self.pending[layer] = (slots, additions)
        heads, queries = q.shape[2], q.shape[1]
        self.rows.append(dict(call=self.calls, layer=layer, current_frame=current_frame, clean_commit=self.clean,
            window_frame_ids=list(frame_ids), candidate_history_tokens=candidate_tokens,
            protected_tokens=len(protected), selected_history_tokens=selected_tokens,
            history_pair_density=selected_tokens/candidate_tokens if candidate_tokens else None,
            actual_attention_tokens=actual_tokens, logical_pairs=heads*queries*actual_tokens,
            full_native_pairs=heads*queries*k.shape[1], route_reused=reuse,
            selection_host_s=selection_host_s, prepare_host_s=prepare_host_s,
            CPU_raw_archive_bytes=0, history_KV_H2D_bytes=0,
            index_H2D_bytes=0 if keep_indices is None or reuse else keep_indices.numel()*keep_indices.element_size(),
            index_select_KV_write_bytes=0 if keep_indices is None else 2*actual_tokens*heads*q.shape[-1]*k.element_size(),
            backend='native_FA2', full_resident_KV_was_already_materialized=True))
        return output

    def attach(self):
        import wan_5b.modules.causal_model as native
        self._native = native
        cls = native.CausalWanSelfAttention
        self._forward = cls.forward
        source = textwrap.dedent(getattr(cls.forward, '_research_source', None) or inspect.getsource(cls.forward))
        tree = ast.parse(source)
        bridge = self
        replaced = 0
        class Rewrite(ast.NodeTransformer):
            def visit_Call(self, node):
                nonlocal replaced
                if (isinstance(node.func, ast.Name) and node.func.id == 'attention'
                    and [getattr(x, 'id', None) for x in node.args] == ['roped_query', 'window_k', 'window_v']):
                    replaced += 1
                    expr = '_research_native_dispatch(self, attention, roped_query, window_k, window_v, info=cache_update_info, current_start=current_start, window_start=window_start, effective_sink=effective_sink, pinned_start=pinned_start_val, pinned_len=pinned_len_val, prepend_sink=prepend_sink, prepend_pinned=prepend_pinned, max_tokens=self.max_attention_size, cache_end=local_end_index, global_sink_tokens=global_sink_tokens)'
                    return ast.copy_location(ast.parse(expr, mode='eval').body, node)
                return self.generic_visit(node)
        tree = Rewrite().visit(tree)
        ast.fix_missing_locations(tree)
        if replaced != 1:
            raise RuntimeError('native absolute Attention dispatch source changed')
        self.derived_source = ast.unparse(tree)
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.derived_sha256 = hashlib.sha256(self.derived_source.encode()).hexdigest()
        layer_ids = {id(block.self_attn): i for i,block in enumerate(self.layers)}
        def dispatch(owner, original, q, k, v, **kwargs):
            if id(owner) not in layer_ids:
                return original(q,k,v)
            return bridge.dispatch(layer_ids[id(owner)], original,q,k,v,**kwargs)
        if hasattr(native, '_research_native_dispatch'):
            raise RuntimeError('a native research bridge is already installed')
        native._research_native_dispatch = dispatch
        scope = {}
        # Preserve the original live globals, including wrapper-published grid
        # and cache scalars. A copied globals dict would retain stale metadata.
        exec(compile(tree, '<native_resident_history_forward>', 'exec'), native.__dict__, scope)
        cls.forward = scope['forward']
        cls.forward._research_source = self.derived_source
        self.handles = [self.pipe.generator.register_forward_pre_hook(self.before, with_kwargs=True),
                        self.pipe.generator.register_forward_hook(self.after, with_kwargs=True)]

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self._forward is not None:
            self._native.CausalWanSelfAttention.forward = self._forward
            if hasattr(self._native, '_research_native_dispatch'):
                del self._native._research_native_dispatch
            self._forward = None

    def audit(self):
        return dict(schema='native_resident_history_v1', config=self.config.__dict__, rows=self.rows,
            upstream_forward_sha256=self.source_sha256, derived_forward_sha256=self.derived_sha256,
            summary_GPU_peak_bytes=self.summary_peak_bytes, summary_build_host_s=self.summary_build_host_s,
            CPU_raw_archive_bytes=0, cross_head_shared_selection=True,
            candidate_scope='native_resident_history_excluding_current_global_sink_and_pinned',
            summary_creation='new_clean_KV_only_committed_after_generator_call',
            proxy_distribution='eligible_history_summaries_only_not_full_attention',
            raw_KV_unchanged=True, positions_preserved=True, offload_path_implemented=False,
            native_window_preparation_not_eliminated=True)
