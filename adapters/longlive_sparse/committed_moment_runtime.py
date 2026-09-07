"""Causal video experiment: archive-time moments plus admitted original KV.

This opt-in wrapper is restricted to upstream_zero RoPE and one shared raw
union. It preserves the raw selector, adds virtual count-weighted tail edges,
and explicitly charges index construction/D2H, moment pack/H2D, subtraction,
and attention. It does not yet bound total CPU history or promote quality.
"""
import time
import torch

from .backends import BackendResult
from .committed_moments import build_frame_moments, gather_moment_payload, subtract_admitted_raw
from .history_cache import tensor_sha256
from .phase_prototypes import archive_rope0_key
from .prototype_reference_runtime import PrototypeReferenceRuntime
from .prototype_tail import execute_weighted_tail_sdpa


class CommittedMomentRuntime(PrototypeReferenceRuntime):
    def __init__(self, pipeline, *, block_tokens=16, grouping='spatial', groups=4, iterations=4):
        super().__init__(pipeline, mode='prototype_tail', block_tokens=block_tokens)
        self.grouping, self.groups, self.iterations = grouping, groups, iterations
        self.frames = {}
        self.epoch = None
        self.frequency_snapshot = None
        self.index_records = []
        self.index_D2H_bytes = 0
        self.moment_H2D_bytes = 0
        self.index_complete_s = 0.
        self.materialize_complete_s = 0.
        self.frequency_validation_s = 0.

    def _epoch(self):
        epoch = self.pipeline.sparse_history_archive.epoch
        if epoch != self.epoch:
            self.frames.clear()
            self.cache.clear()
            self.frequency_snapshot = None
            self.epoch = epoch

    def _frequencies(self, freqs):
        begin = time.perf_counter()
        if self.frequency_snapshot is None:
            self.frequency_snapshot = freqs.detach().clone()
        elif not torch.equal(freqs, self.frequency_snapshot):
            raise RuntimeError('RoPE table changed after committing moments; restart archive epoch')
        if freqs.is_cuda:
            torch.cuda.current_stream(freqs.device).synchronize()
        self.frequency_validation_s += time.perf_counter()-begin

    def __enter__(self):
        archive = self.pipeline.sparse_history_archive
        if archive.config.rope_policy != 'upstream_zero':
            raise ValueError('committed moments currently require invariant upstream_zero RoPE')
        super().__enter__()
        reference_route = archive.route_indexed
        def route(*args, **kwargs):
            plan = reference_route(*args, **kwargs)
            plan.method = 'committed_moment_' + self.grouping
            plan.metadata['routing_identity'].update(
                reference_full_candidate_materialization=False, moment_storage='FP32_additive_sums',
                moment_grouping=self.grouping, moment_groups=self.groups, moment_iterations=self.iterations)
            return plan
        archive.route_indexed = route
        self.original_index = archive.index_frame
        def index_frame(layer_id, frame_id, key, value, **kwargs):
            self._epoch()
            if self.active is None:
                raise RuntimeError('commit moments require the actual active module RoPE table')
            module, arguments = self.active
            if int(layer_id) != module.layer_id:
                raise RuntimeError('archive/module layer mismatch')
            freqs = arguments['freqs']
            self._frequencies(freqs)
            if key.device != freqs.device:
                raise RuntimeError('construct moments from the already resident evicted frame, not onloaded old raw KV')
            indexed = self.original_index(layer_id, frame_id, key, value, **kwargs)
            begin = time.perf_counter()
            rotated = archive_rope0_key(key, spatial_height=archive.spatial_height,
                spatial_width=archive.spatial_width, freqs=freqs, rope_policy=archive.config.rope_policy)
            moment = build_frame_moments(rotated, value, block_tokens=self.block_tokens,
                grouping=self.grouping, groups=self.groups, iterations=self.iterations).cpu()
            if key.is_cuda:
                torch.cuda.current_stream(key.device).synchronize()
            elapsed = time.perf_counter()-begin
            version = archive.frame_storage_version(layer_id,frame_id)
            self.frames[(int(layer_id),int(frame_id))] = (version,moment)
            self.index_D2H_bytes += moment.bytes
            self.index_complete_s += elapsed
            self.index_records.append(dict(layer=int(layer_id),frame=int(frame_id),epoch=archive.epoch,
                storage_version=version,bytes=moment.bytes,complete_s=elapsed))
            return indexed
        archive.index_frame = index_frame
        return self

    def execute(self, backend, query, exact_key, exact_value, history_key, history_value, plan, **unused):
        self._epoch()
        if self.active is None or plan.groups != 1:
            raise ValueError('committed moment experiment requires one shared raw group')
        module, arguments = self.active
        archive = self.pipeline.sparse_history_archive
        self._frequencies(arguments['freqs'])
        start = int(arguments.get('current_start',0))
        ids = tuple(int(x) for x in (arguments['memory_indices'][0].long()+int(module.sink_size)).cpu())
        raw_sha = plan.digest()
        versions = tuple(archive.frame_storage_version(module.layer_id,i) for i in ids)
        key = (archive.epoch,archive.layer_storage_version(module.layer_id),start,ids,versions,raw_sha,
               str(query.dtype),str(query.device),tuple(query.shape),tuple(query.stride()),
               self.block_tokens,self.grouping,self.groups,self.iterations,archive.config.rope_policy)
        hit = module.layer_id in self.cache and self.cache[module.layer_id][0] == key
        if hit:
            tail = self.cache[module.layer_id][1]
        else:
            begin = time.perf_counter()
            frames = []
            for frame_id, version in zip(ids,versions):
                recorded = self.frames.get((module.layer_id,frame_id))
                if recorded is None or recorded[0] != version:
                    raise RuntimeError('missing/stale committed moment; no full-candidate fallback')
                frames.append(recorded[1])
            payload = gather_moment_payload(frames,ids,plan.union_frame_ids,plan.union_token_ids)
            self.moment_H2D_bytes += sum(t.numel()*t.element_size() for t in payload)
            payload = tuple(t.to(query.device) for t in payload)
            tail = subtract_admitted_raw(payload,history_key,history_value,
                frame_tokens=archive.spatial_height*archive.spatial_width,block_tokens=self.block_tokens,
                prototype_dtype=query.dtype)
            torch.cuda.current_stream(query.device).synchronize()
            self.materialize_complete_s += time.perf_counter()-begin
            self.cache[module.layer_id] = (key,tail)
        begin = time.perf_counter()
        output = execute_weighted_tail_sdpa(query,exact_key,exact_value,history_key,history_value,tail)
        torch.cuda.current_stream(query.device).synchronize()
        elapsed = time.perf_counter()-begin
        pairs = query.shape[0]*query.shape[1]*query.shape[2]*(exact_key.shape[1]+history_key.shape[1]+tail.key.shape[1])
        logical = query.shape[0]*query.shape[1]*query.shape[2]*(exact_key.shape[1]+history_key.shape[1])+query.shape[1]*int((tail.counts>0).sum())
        self.records.append(dict(layer=module.layer_id,current_start=start,raw_route_sha256=raw_sha,
            raw_coordinates_sha256=tensor_sha256(torch.stack((plan.union_frame_ids,plan.union_token_ids),-1)),
            prototype_count_sha256=tensor_sha256(tail.counts),prototype_slots=tail.key.shape[1],
            prototype_cache_hit=hit,weighted_operator_s=elapsed,physical_virtual_pairs=pairs))
        return BackendResult(output,'committed_moment_memory_efficient_sdpa',elapsed*1000,logical,pairs,pairs-logical,raw_sha)

    def __exit__(self, exc_type, exc_value, tb):
        self.pipeline.sparse_history_archive.index_frame = self.original_index
        return super().__exit__(exc_type,exc_value,tb)

    def audit(self):
        return dict(mode='committed_moment',grouping=self.grouping,groups=self.groups,block_tokens=self.block_tokens,
            calls=len(self.records),records=self.records,index_records=self.index_records,
            extra_candidate_H2D_bytes=0,extra_metadata_H2D_bytes=self.moment_H2D_bytes,
            extra_index_D2H_bytes=self.index_D2H_bytes,index_complete_s=self.index_complete_s,
            materialize_complete_s=self.materialize_complete_s,frequency_validation_s=self.frequency_validation_s,
            total_tail_build_s=self.index_complete_s+self.materialize_complete_s+self.frequency_validation_s,
            CPU_moment_storage_bytes=sum(moment.bytes for _,moment in self.frames.values()),
            resident_prototype_bytes=sum(tail.bytes for _,tail in self.cache.values()),
            online_selector_full_candidate_access=False,future_video_access=False,
            complete_candidate_read_after_raw_selection=False,raw_route_does_not_describe_virtual_prototype_edges=True,
            bounded_CPU_history=False,wall_speedup_claim=False)
