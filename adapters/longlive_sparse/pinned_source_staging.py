"""One bounded pinned layer for a capacity-limited source reader.

CPU packing is charged. This is not prefetch and retains repeated layer H2D.
The existing reader synchronizes rephase completion before reusing this slot.
"""
import time

import torch


class PinnedSourceStagingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_pinned_pair = None
        self.source_pinned_stats = dict(calls=0, CPU_pack_bytes=0, CPU_pack_wall_s=0.,
            pinned_allocation_s=0., pinned_owned_bytes=0, pinned_budget_bytes=96*1024**2)

    def copy_source_pair(self, cpu_pair, device):
        if len(cpu_pair)!=2 or any(t.device.type!='cpu' or t.dtype!=torch.bfloat16 for t in cpu_pair):
            raise ValueError('two BF16 CPU source tensors required')
        if cpu_pair[0].shape!=cpu_pair[1].shape:
            raise ValueError('K/V source shapes differ')
        size=sum(t.numel()*t.element_size() for t in cpu_pair)
        stats=self.source_pinned_stats
        if size>stats['pinned_budget_bytes']:
            raise ValueError('one-layer source exceeds fixed96MiB pinned budget')
        if self.source_pinned_pair is None:
            began=time.perf_counter()
            self.source_pinned_pair=tuple(torch.empty_like(t,device='cpu',pin_memory=True) for t in cpu_pair)
            stats.update(pinned_owned_bytes=size,pinned_allocation_s=time.perf_counter()-began)
        if any(p.shape!=s.shape for p,s in zip(self.source_pinned_pair,cpu_pair)):
            raise ValueError('source geometry changed; do not silently resize pinned workspace')
        began=time.perf_counter()
        for pinned,source in zip(self.source_pinned_pair,cpu_pair):pinned.copy_(source)
        stats['CPU_pack_wall_s']+=time.perf_counter()-began
        stats['CPU_pack_bytes']+=size;stats['calls']+=1
        # The caller's existing event2.synchronize completes both transfers
        # before another layer can overwrite the single pinned slot.
        return tuple(p.to(device=device,copy=True,non_blocking=True) for p in self.source_pinned_pair)

    def audit(self):
        result=super().audit()
        result['source_pinned_staging']=dict(self.source_pinned_stats,
            prefetch=False,source_H2D_reduced=False,CPU_archive_unchanged=True,
            pinned_reuse_fence='existing source rephase completion synchronization',
            pack_is_counted_in_existing_load_wall=True,source_copy_stream_span_includes_CPU_pack_gap=True)
        return result
