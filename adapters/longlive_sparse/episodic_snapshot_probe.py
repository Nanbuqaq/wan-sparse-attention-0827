"""Bounded, privileged KV-version intervention, NOT an online retrieval method.

Keep a fixed number of completed raw KV frames at a declared episode boundary.
On a declared return, replace equally many non-sink local frames. The original
local cache applies its normal relative RoPE and eviction afterwards. Thus the
experiment changes memory content/version, not Attention size or local layout.
"""
from dataclasses import dataclass

import torch

from .history_cache import tensor_sha256


@dataclass(frozen=True)
class SnapshotLayer:
    layer: int
    key: torch.Tensor
    value: torch.Tensor
    global_frames: tuple[int, ...]


class EpisodicSnapshotProbe:
    def __init__(self, *, frames=6, frame_tokens=1560, sink_frames=1,
                 next_chunk_frames=3, budget_bytes=2 * 1024**3):
        if min(frames, frame_tokens, next_chunk_frames, budget_bytes) <= 0 or sink_frames < 0:
            raise ValueError('invalid snapshot geometry/budget')
        self.frames, self.frame_tokens, self.sink_frames = frames, frame_tokens, sink_frames
        self.next_chunk_frames, self.budget_bytes = next_chunk_frames, budget_bytes
        self.layers = ()
        self.capture_event = None
        self.restore_event = None

    def capture(self, caches, *, completed_frames, version):
        if self.layers or not caches:
            raise ValueError('one nonempty episode capture per probe')
        count = self.frames * self.frame_tokens
        required = 0
        for cache in caches:
            end = int(cache['local_end_index'])
            if end // self.frame_tokens > completed_frames or end % self.frame_tokens:
                raise ValueError('uncommitted or unaligned cache')
            if end - count < self.sink_frames * self.frame_tokens:
                raise ValueError('episode cannot include sink or unavailable frames')
            if cache['k'].shape != cache['v'].shape or cache['k'].ndim != 4:
                raise ValueError('expected matching raw [B,T,H,D] K/V')
            required += count * cache['k'].shape[0] * cache['k'].shape[2] * cache['k'].shape[3] * (
                cache['k'].element_size() + cache['v'].element_size())
        if required > self.budget_bytes:
            raise ValueError('bounded episode budget exceeded')
        frames = tuple(range(completed_frames-self.frames, completed_frames))
        self.layers = tuple(SnapshotLayer(i,
            cache['k'][:,int(cache['local_end_index'])-count:int(cache['local_end_index'])].detach().to('cpu',copy=True).contiguous(),
            cache['v'][:,int(cache['local_end_index'])-count:int(cache['local_end_index'])].detach().to('cpu',copy=True).contiguous(),
            frames) for i,cache in enumerate(caches))
        self.capture_event = dict(completed_frames=completed_frames,version=version,global_frames=list(frames),
            raw_KV_bytes=required,CPU_archive_peak_bytes=required,budget_bytes=self.budget_bytes,
            D2H_bytes=required if caches[0]['k'].is_cuda else 0,
            source_layer_SHA256=[dict(layer=s.layer,K=tensor_sha256(s.key),V=tensor_sha256(s.value)) for s in self.layers])
        return self.capture_event

    def restore(self, caches, *, current_start):
        if not self.layers or self.restore_event is not None or len(caches)!=len(self.layers):
            raise ValueError('one restore of the complete captured episode required')
        if current_start <= self.capture_event['completed_frames']:
            raise ValueError('restore must be strictly later than committed episode')
        count=self.frames*self.frame_tokens
        plans=[]
        for snapshot,cache in zip(self.layers,caches):
            end=int(cache['local_end_index']); capacity=cache['k'].shape[1]
            evicted=max(0,end+self.next_chunk_frames*self.frame_tokens-capacity)
            start=self.sink_frames*self.frame_tokens+evicted
            if start+count>end or snapshot.key.shape[0:1]+snapshot.key.shape[2:] != cache['k'].shape[0:1]+cache['k'].shape[2:]:
                raise ValueError('episode does not fit surviving non-sink local context')
            if cache['k'].dtype!=snapshot.key.dtype or cache['v'].dtype!=snapshot.value.dtype:
                raise ValueError('snapshot dtype mismatch')
            plans.append((snapshot,cache,start,start+count))
        for snapshot,cache,start,end in plans:
            cache['k'][:,start:end].copy_(snapshot.key)
            cache['v'][:,start:end].copy_(snapshot.value)
        self.restore_event=dict(current_start=current_start,source_global_frames=list(self.layers[0].global_frames),
            raw_KV_bytes=self.capture_event['raw_KV_bytes'],
            H2D_bytes=self.capture_event['raw_KV_bytes'] if caches[0]['k'].is_cuda else 0,
            local_slots_before_next_roll=[dict(layer=s.layer,start_token=a,end_token=b) for s,c,a,b in plans],
            position_policy='raw_KV_repositioned_to_existing_local_slots_before_normal_RoPE',
            global_and_local_end_indices_unchanged=True,attention_size_unchanged=True,
            retention_policy='one_restore_then_native_eviction_no_persistent_reinjection')
        return self.restore_event
