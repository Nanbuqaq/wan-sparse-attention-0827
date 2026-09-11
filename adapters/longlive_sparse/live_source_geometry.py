"""Bounded geometry worker and source-version-checked native memory bridge."""
import threading
import time
import traceback

import torch

from .bounded_worker import BoundedWorker
from .native_causal_block_memory import NativeCausalBlockMemory
from .native_oracle_source_mask import fixed_mask_indices


class SourceGeometryWorker:
    def __init__(self,model,*,started,slots=2,max_sources=3,component_policy='mutual_geometry'):
        self.model=model;self.started=started;self.max_sources=max_sources;self.policy=component_policy
        self.condition=threading.Condition();self.submitted={};self.results={};self.waits=[];self.closed=False
        self.worker=BoundedWorker(slots,self._consume,name='source-geometry-worker')

    def submit(self,window):
        version=window['archive_version'];began=time.perf_counter()
        with self.condition:
            if version in self.submitted or len(self.submitted)>=self.max_sources:
                raise ValueError('unique source versions within the registered cap required')
            self.submitted[version]=dict(archive_version=version,source_end=window['source_end'],
                ready_s=window['ready_s'],submit_started_s=began-self.started)
        slot=self.worker.reserve()
        try:self.worker.enqueue(slot,(window,time.perf_counter()-self.started))
        except BaseException:self.worker.release_unsubmitted(slot);raise

    def _consume(self,slot,job):
        window,enqueued=job;began=time.perf_counter();version=window['archive_version']
        try:
            result=self.model.extract(window,component_policy=self.policy,keep_pixel_masks=False)
            result.update(source_phase=window['source_phase'],pixel_start=window['pixel_start'],pixel_end=window['pixel_end'])
        except BaseException as error:
            result=dict(status='geometry_error',archive_version=version,source_end=window['source_end'],
                error=repr(error),traceback=traceback.format_exc())
        result.update(enqueued_s=enqueued,work_started_s=began-self.started,
            work_finished_s=time.perf_counter()-self.started,work_wall_s=time.perf_counter()-began)
        with self.condition:self.results[version]=result;self.condition.notify_all()

    def get(self,version,timeout=60.):
        began=time.perf_counter();deadline=began+timeout
        with self.condition:
            while version not in self.results:
                remaining=deadline-time.perf_counter()
                if remaining<=0:raise RuntimeError('selected source geometry readiness timeout')
                self.condition.wait(min(.1,remaining))
            result=self.results[version]
            self.waits.append(dict(archive_version=version,requested_s=began-self.started,
                returned_s=time.perf_counter()-self.started,wait_s=time.perf_counter()-began))
        if result['status']!='mask_ready':raise RuntimeError('selected source geometry failed: '+result.get('error','unknown'))
        return result

    def finish(self):
        if not self.closed:self.worker.finish();self.closed=True

    def abort(self):
        if not self.closed:self.worker.abort();self.closed=True

    def audit(self):
        with self.condition:
            rows=[]
            for version,r in sorted(self.results.items()):
                row={k:v for k,v in r.items() if not isinstance(v,torch.Tensor)}
                row['source_tokens']=r['indices'].numel() if 'indices' in r else None;rows.append(row)
            return dict(rows=rows,submitted=list(self.submitted.values()),selected_waits=list(self.waits),
                producer_backpressure_s=self.worker.backpressure_s,max_sources=self.max_sources,
                retained_result_tensor_bytes=sum(v.numel()*v.element_size() for r in self.results.values() for v in r.values() if isinstance(v,torch.Tensor)),
                unselected_source_errors_do_not_abort_other_sources=True,
                no_dense_or_manual_fallback=True,source_pixels_are_current_run_pre_codec=True)

    def export(self,path):
        with self.condition:
            with path.open('xb') as f:torch.save(dict(schema='live_source_geometry_v1',results=self.results),f)


class LiveSourceGeometryMemory(NativeCausalBlockMemory):
    def __init__(self,pipe,config,token_grid,*,mask_fill='uniform_midpoint'):
        if config.policy!='source_mask' or config.source_repeats!=1:raise ValueError('live source geometry requires the mask bridge')
        if mask_fill not in ('uniform_midpoint','fixed_bit_reversal'):raise ValueError('unknown background fill')
        self.mask_fill=mask_fill
        super().__init__(pipe,config,token_grid)
        self.geometry_worker=None;self.geometry_plans={};self.geometry_used_layers=set();self.geometry_sources=[]

    def mask_source_indices(self,layer,heads,selected):
        descriptor=self.active_bank['descriptor'];version=descriptor.archive_version;key=(version,selected)
        if key not in self.geometry_plans:
            if self.geometry_worker is None:raise RuntimeError('live geometry producer was not attached')
            began=time.perf_counter();r=self.geometry_worker.get(version)
            self.ledger['live_geometry_ready_wait_s']=self.ledger.get('live_geometry_ready_wait_s',0.)+time.perf_counter()-began
            if (r['source_end']!=descriptor.source_end or r['source_start']!=descriptor.source_end-8
                or r['source_phase']!=descriptor.source_phase or not r['all_masks_nonempty']
                or r['pixel_input_kind']!='raw_stream_before_codec'
                or tuple(r['token_masks'].shape)!=(8,*self.grid)
                or not torch.equal(r['indices'],torch.where(r['token_masks'].flatten())[0])):
                raise RuntimeError('live geometry ownership or shape differs from selected source')
            if self.mask_fill=='fixed_bit_reversal':
                from .stable_source_mask_fill import stable_mask_indices
                self.geometry_plans[key]=stable_mask_indices(r['indices'],source_tokens=8*self.frame_tokens,budget=selected)
            else:
                self.geometry_plans[key]=fixed_mask_indices(r['indices'],source_tokens=8*self.frame_tokens,budget=selected,mode='foreground')
            self.geometry_sources.append(dict(archive_version=version,source_end=r['source_end'],
                source_latent_sha256=r['source_latent_sha256'],source_raw_pixel_sha256=r['source_raw_pixel_sha256'],
                mask_tokens=r['indices'].numel(),selected_tokens=selected))
        self.geometry_used_layers.add(layer)
        return self.geometry_plans[key][None].repeat(heads,1)

    def audit(self):
        result=super().audit();result.update(method_variant='live_source_geometry',oracle=False,
            background_fill=self.mask_fill,
            automatic_online_method=True,semantic_target_selection=False,geometry_sources=self.geometry_sources,
            geometry_used_layers=sorted(self.geometry_used_layers),
            geometry_plan_CPU_bytes=sum(x.numel()*x.element_size() for x in self.geometry_plans.values()),
            scope='qualified three-archive native32 prototype; not arbitrary archive growth or a quality winner')
        return result
