"""Resident-model paired CPU output experiment; no offline codec audit on GPU."""
import json
import gc
import os
import resource
import time
from pathlib import Path


def schedule(pairs=4):
    if pairs < 3:
        raise ValueError('at least three paired repetitions required')
    rows=[dict(id='warmup_inline',mode='inline',measured=False),
          dict(id='warmup_thread',mode='thread',measured=False)]
    for pair in range(pairs):
        for mode in (('inline','thread') if pair%2==0 else ('thread','inline')):
            rows.append(dict(id=f'pair{pair}_{mode}',mode=mode,measured=True,pair=pair))
    return rows


class FileBarrier:
    """Fresh batch directory, per-round atomic records, deadline and peer failure."""
    def __init__(self,root,lane,lanes=4,timeout=180):
        self.root=Path(root);self.lane=lane;self.lanes=lanes;self.timeout=timeout
        if not 0 <= lane < lanes:raise ValueError('invalid lane')
        self.root.mkdir(parents=True,exist_ok=True)

    def fail(self,error):
        p=self.root/f'failed_lane{self.lane}.json'
        p.write_text(json.dumps(dict(lane=self.lane,error=str(error))))

    def wait(self,phase):
        if '/' in phase or '..' in phase:raise ValueError('invalid phase')
        p=self.root/phase;p.mkdir(exist_ok=True)
        tmp=p/f'lane{self.lane}.tmp';ready=p/f'lane{self.lane}.json'
        if ready.exists():raise RuntimeError('stale or reused barrier round')
        tmp.write_text(json.dumps(dict(lane=self.lane,ready_ns=time.monotonic_ns())))
        tmp.replace(ready);deadline=time.monotonic()+self.timeout
        while True:
            if list(self.root.glob('failed_lane*.json')):raise RuntimeError('peer failed')
            if all((p/f'lane{i}.json').is_file() for i in range(self.lanes)):
                # Common future epoch based on the last participant, all on one host.
                target=max(json.loads((p/f'lane{i}.json').read_text())['ready_ns'] for i in range(self.lanes))+200_000_000
                while time.monotonic_ns()<target:time.sleep(.001)
                return dict(target_ns=target,released_ns=time.monotonic_ns())
            if time.monotonic()>deadline:raise TimeoutError(f'barrier timeout: {phase}')
            time.sleep(.02)


def resource_snapshot():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('Threads','VmRSS','VmHWM','VmPin','VmLck','Cpus_allowed_list','Mems_allowed_list'):
            status[key]=value.strip()
    return dict(monotonic_ns=time.monotonic_ns(),user_cpu_s=usage.ru_utime,
                system_cpu_s=usage.ru_stime,max_rss_kib=usage.ru_maxrss,proc_status=status)


def run(pipe,noise,prompts,report,output,barrier_root,lane,pairs=4):
    import torch
    from wan_5b.modules.vae2_2 import unpatchify
    from .native_video_pipeline import NativeVideoPipeline
    from .stream_video_sink import IncrementalVideoSink
    from .history_cache import tensor_sha256
    from scripts.audit_native_pixel_completion import validate_groups
    barrier=FileBarrier(barrier_root,lane,timeout=900)
    rows=[];reference_latent=None;reference_rgb=None;pipeline=None
    try:
        cases=schedule(pairs)+[dict(id='diagnostic_'+m,mode=m,measured=False,diagnostic=True) for m in ('inline','thread')]
        for case in cases:
            directory=Path(output)/case['id'];directory.mkdir(exist_ok=False)
            # Clear all KV and cross-attention contents, not just offsets. Identical
            # fresh allocation is charged inside the measured inference in each arm.
            pipe.clear_cache()
            torch.manual_seed(report['seed']);torch.cuda.manual_seed_all(report['seed'])
            torch.cuda.synchronize(0);torch.cuda.synchronize(1)
            torch.cuda.reset_peak_memory_stats(0);torch.cuda.reset_peak_memory_stats(1)
            release=barrier.wait(case['id']+'_start')
            resources_before=resource_snapshot()
            started=time.perf_counter();start_ns=time.monotonic_ns()
            sink=IncrementalVideoSink(directory/'video.mp4',expected_frames=509,started=started,fps=24,
                measure_stages=case.get('diagnostic',False))
            pipeline=NativeVideoPipeline(pipe.vae,unpatchify,sink,source_device='cuda:0',target_device='cuda:1',
                latent_shape=list(noise.shape),started=started,slots=2,pinned_budget=128*1024**2,
                serial=False,encode_mode=case['mode'],pixel_slots=2)
            pipeline.attach(pipe)
            with torch.inference_mode():latent=pipe.inference(noise=noise,text_prompts=prompts,return_latents=True)
            torch.cuda.synchronize(0);generation_s=time.perf_counter()-started
            pipeline.detach();pixels,ledger=pipeline.finish(generation_finished_s=generation_s)
            delivery_ns=time.monotonic_ns()
            resources_after=resource_snapshot()
            # Finish barrier keeps correctness/hash/I/O out of peers' timing windows.
            barrier.wait(case['id']+'_delivered')
            pipeline.write_trace(directory/'pipeline_host_trace.json')
            if case.get('diagnostic'):
                (directory/'cpu_stages.json').write_text(json.dumps(dict(records=sink.stage_records,
                    scope='host wall and calling thread CPU; excludes libx264 worker CPU',
                    measured_timing_case=False),indent=2))
            cpu=latent.cpu();latent_hash=tensor_sha256(cpu);validate_groups(ledger)
            if reference_latent is None:
                reference_latent=cpu.clone();reference_rgb=pixels['raw_RGB_sha256']
            if not torch.equal(cpu,reference_latent):raise RuntimeError('full latent mismatch')
            if pixels['raw_RGB_sha256']!=reference_rgb:raise RuntimeError('raw RGB mismatch')
            if ledger['streamed_latent_sha256']!=latent_hash:raise RuntimeError('streamed latent mismatch')
            torch.save(cpu,directory/'latents.pt')
            summary=dict(report,status='pass',case=case,measured=case['measured'],barrier=release,
                monotonic_start_ns=start_ns,monotonic_delivery_ns=delivery_ns,pixels=pixels,video_pipeline=ledger,
                latent_sha256=latent_hash,full_latent_equal_to_first_warmup=True,
                generation_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                pipeline_VAE_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(1),
                model_resident=True,MP4_decode_audit_pending=True,
                resources_before=resources_before,resources_after=resources_after,
                process_CPU_seconds=(resources_after['user_cpu_s']+resources_after['system_cpu_s']-
                    resources_before['user_cpu_s']-resources_before['system_cpu_s']))
            (directory/'summary.json').write_text(json.dumps(summary,indent=2))
            rows.append(dict(case=case,path=str(directory),complete_s=ledger['complete_s']))
            del pipeline,latent,cpu,sink
            gc.collect()
            torch.cuda.empty_cache()
            pipeline=None
        return dict(status='pass',cases=rows,model_loads=1,measured_pairs=pairs)
    except BaseException as error:
        barrier.fail(error)
        if pipeline is not None:pipeline.abort()
        raise
