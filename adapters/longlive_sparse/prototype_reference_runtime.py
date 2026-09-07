"""Process-local causal prototype-tail reference for video mechanism tests.

The base selector still uses compact online inputs. This reference then reads
and uploads the COMPLETE selected coarse candidate set to construct a tail.
That overhead is charged; this is not an efficient onload implementation.
"""
from contextlib import AbstractContextManager
import inspect
import time

import torch

from .backends import BackendResult
from .history_cache import tensor_sha256
from .prototype_tail import PrototypeTail, build_prototype_tail, execute_weighted_tail_sdpa
from .rope import apply_selected_rope, build_sparse_positions
from .route_plan import map_union_coordinates


class PrototypeReferenceRuntime(AbstractContextManager):
    def __init__(self, pipeline, *, mode, block_tokens=64):
        if mode not in ('sdpa_null', 'prototype_tail'):
            raise ValueError('unknown reference operator')
        self.pipeline, self.mode, self.block_tokens = pipeline, mode, block_tokens
        self.active = None
        self.cache, self.records, self.hooks = {}, [], []
        self.extra_candidate_H2D_bytes = 0
        self.extra_metadata_H2D_bytes = 0
        self.tail_build_s = 0.

    def __enter__(self):
        from . import runtime_attention as runtime
        self.runtime = runtime
        self.original_execute = runtime.execute_plan
        self.original_route = self.pipeline.sparse_history_archive.route_indexed
        def route(*args, **kwargs):
            plan = self.original_route(*args, **kwargs)
            previous = plan.metadata.get('routing_identity')
            plan.method = 'prototype_reference_' + self.mode
            plan.metadata['routing_identity'] = {'base_identity': previous, 'operator': self.mode,
                'tail_block_tokens': self.block_tokens, 'prototype_dtype': 'BF16',
                'reference_full_candidate_materialization': self.mode == 'prototype_tail'}
            return plan
        self.pipeline.sparse_history_archive.route_indexed = route
        for module in self.pipeline.sparse_history_modules:
            signature = inspect.signature(module.forward)
            def before(owner, args, kwargs, signature=signature):
                bound = signature.bind_partial(*args, **kwargs).arguments
                self.active = (owner, bound)
            self.hooks.append(module.register_forward_pre_hook(before, with_kwargs=True))
        runtime.execute_plan = self.execute
        return self

    def execute(self, backend, query, exact_key, exact_value, history_key, history_value, plan, **unused):
        if self.active is None or plan.groups != 1:
            raise ValueError('reference requires an active module and shared raw union')
        module, arguments = self.active
        start = int(arguments.get('current_start', 0))
        archive = self.pipeline.sparse_history_archive
        ids = arguments['memory_indices'][0].long()+int(module.sink_size)
        candidate_ids = tuple(int(x) for x in ids.cpu())
        raw_sha = plan.digest()
        cache_key = (archive.epoch, archive.layer_storage_version(module.layer_id), start,
                     candidate_ids, raw_sha, str(query.dtype), str(query.device), self.block_tokens,
                     module.sparse_config.rope_policy, module.sparse_config.max_relative_age,
                     id(arguments['freqs']))
        if self.mode == 'sdpa_null':
            tail = PrototypeTail(history_key[:, :0], history_value[:, :0],
                torch.empty(query.shape[0], query.shape[2], 0, device=query.device), self.block_tokens,
                archive.spatial_height*archive.spatial_width)
            cache_hit = False
        elif module.layer_id in self.cache and self.cache[module.layer_id][0] == cache_key:
            tail = self.cache[module.layer_id][1]
            cache_hit = True
        else:
            begin = time.perf_counter()
            raw_key, raw_value, frame_ids, token_ids = archive.dense_history_tensors(module.layer_id, ids)
            positions = build_sparse_positions(frame_ids=frame_ids, token_ids=token_ids,
                current_frame_id=start//(archive.spatial_height*archive.spatial_width),
                spatial_width=archive.spatial_width, rope_policy=module.sparse_config.rope_policy,
                max_relative_age=module.sparse_config.max_relative_age, candidate_frame_ids=ids)
            indices = map_union_coordinates(plan, frame_ids, token_ids)
            full_key, full_value = raw_key.to(query.device), raw_value.to(query.device)
            key = apply_selected_rope(full_key, positions.to(query.device), arguments['freqs']).to(query.dtype)
            tail = build_prototype_tail(key, full_value, indices,
                frame_tokens=archive.spatial_height*archive.spatial_width, block_tokens=self.block_tokens)
            torch.cuda.current_stream(query.device).synchronize()
            self.tail_build_s += time.perf_counter()-begin
            self.extra_candidate_H2D_bytes += sum(x.numel()*x.element_size() for x in (raw_key, raw_value))
            self.extra_metadata_H2D_bytes += sum(x.numel()*x.element_size() for x in (positions, indices))
            self.cache[module.layer_id] = (cache_key, tail)
            cache_hit = False
        begin = time.perf_counter()
        output = execute_weighted_tail_sdpa(query, exact_key, exact_value, history_key, history_value, tail)
        torch.cuda.current_stream(query.device).synchronize()
        elapsed = time.perf_counter()-begin
        pairs = query.shape[0]*query.shape[1]*query.shape[2]*(exact_key.shape[1]+history_key.shape[1]+tail.key.shape[1])
        logical_pairs = (query.shape[0]*query.shape[1]*query.shape[2]*(exact_key.shape[1]+history_key.shape[1])
                         + query.shape[1]*int((tail.counts > 0).sum()))
        self.records.append(dict(layer=module.layer_id, current_start=start, raw_route_sha256=raw_sha,
            raw_coordinates_sha256=tensor_sha256(torch.stack((plan.union_frame_ids, plan.union_token_ids), -1)),
            prototype_count_sha256=tensor_sha256(tail.counts), prototype_slots=tail.key.shape[1],
            prototype_cache_hit=cache_hit, weighted_operator_s=elapsed, physical_virtual_pairs=pairs))
        return BackendResult(output, 'prototype_'+self.mode+'_memory_efficient_sdpa', elapsed*1000,
                             logical_pairs, pairs, pairs-logical_pairs, raw_sha)

    def __exit__(self, exc_type, exc_value, tb):
        self.runtime.execute_plan = self.original_execute
        self.pipeline.sparse_history_archive.route_indexed = self.original_route
        for hook in self.hooks:
            hook.remove()
        return False

    def audit(self):
        return dict(mode=self.mode, block_tokens=self.block_tokens, calls=len(self.records), records=self.records,
            extra_candidate_H2D_bytes=self.extra_candidate_H2D_bytes,
            extra_metadata_H2D_bytes=self.extra_metadata_H2D_bytes,
            total_tail_build_s=self.tail_build_s,
            resident_prototype_bytes=sum(value.bytes for _, value in self.cache.values()),
            online_selector_full_candidate_access=False, future_video_access=False,
            complete_candidate_read_after_raw_selection=True, efficient_onload=False,
            raw_route_does_not_describe_virtual_prototype_edges=True)
