"""Bounded inference profiling; normal samplers and parameter values are reused.

This is a separate profiled trial. Stream event times, CPU dispatch times,
decoded-frame availability and completed-file delivery are different fields.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import os
import re
import sys
import time


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


class ProfileBook:
    def __init__(self,out,rank,start_forward,forward_count,engine_factory,
                 scope_factory=nullcontext,event_factory=None,capture_vae=True):
        if start_forward<1 or not 1<=forward_count<=8:
            raise ValueError('Use a positive forward index and a bounded1..8 forward capture')
        self.out=Path(out);self.rank=rank;self.start=start_forward;self.count=forward_count
        self.factory=engine_factory;self.scope=scope_factory;self.event_factory=event_factory
        self.capture_vae=capture_vae;self.active=None;self.engine=None;self.completed={}
        self.forward_index=0;self.calls=[];self.events=[];self.host=[];self.started=time.monotonic()
        self.first_decoded_frames_sec=None;self.first_complete_file_sec=None
        self.hooks=[];self.open_scopes=[]
        self.enabled=True

    def begin(self,name):
        if self.engine is not None:raise RuntimeError('Profiler windows must not overlap')
        self.active=name;self.engine=self.factory();self.engine.start()

    def finish(self):
        if self.engine is None:return
        name=self.active;engine=self.engine;self.active=None;self.engine=None
        while self.open_scopes:
            self.open_scopes.pop().__exit__(None,None,None)
        for h in self.hooks:h.remove()
        self.hooks=[];engine.stop();self.out.mkdir(parents=True,exist_ok=True)
        trace=self.out/f'rank{self.rank}_{name}.trace.json';engine.export_chrome_trace(str(trace))
        operators=[]
        for e in engine.key_averages(group_by_input_shape=True):
            operators.append(dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                                  total_cpu_us=e.cpu_time_total,
                                  self_device_us=getattr(e,'self_device_time_total',getattr(e,'self_cuda_time_total',None)),
                                  total_device_us=getattr(e,'device_time_total',getattr(e,'cuda_time_total',None)),
                                  self_device_memory_bytes=getattr(e,'self_device_memory_usage',None),
                                  input_shapes=str(getattr(e,'input_shapes',None))))
        path=self.out/f'rank{self.rank}_{name}.operators.json';atomic_json(path,operators)
        self.completed[name]=dict(trace=str(trace),operators=str(path),operator_count=len(operators),
                                  exclusive_and_inclusive_times_not_additive=True)

    def bind_scopes(self,model):
        if not hasattr(model,'named_modules'):return
        for name,module in model.named_modules():
            low=name.lower()
            if not name:continue
            if re.search(r'(to_q|to_k|to_v|q_proj|k_proj|v_proj|qkv|proj_q|proj_k|proj_v)$',low):category='qkv'
            elif re.search(r'(attn\w*|attention)$',low):category='attention'
            elif re.search(r'(ff\w*|feed_forward)$',low):category='ffn'
            elif re.search(r'(norm\w*|rope|rotary\w*)$',low):category='norm_rope'
            else:continue
            def before(mod,args,label='h3/'+category+'/'+name):
                ctx=self.scope(label);ctx.__enter__();self.open_scopes.append(ctx)
            def after(mod,args,result):
                if self.open_scopes:self.open_scopes.pop().__exit__(None,None,None)
            self.hooks.append(module.register_forward_pre_hook(before))
            self.hooks.append(module.register_forward_hook(after))

    def call_model(self,model,*args,**kwargs):
        self.forward_index+=1;number=self.forward_index
        if number==self.start:
            self.begin('dit');self.bind_scopes(model)
        start=end=None
        if self.event_factory:
            start=self.event_factory();end=self.event_factory();start.record()
        began=time.monotonic()
        try:
            with self.scope('h3/dit_forward') if self.active=='dit' else nullcontext():
                result=model(*args,**kwargs)
        finally:
            elapsed=time.monotonic()-began
            if end is not None:end.record();self.events.append((number,start,end))
            self.calls.append(dict(forward=number,host_dispatch_wall_sec=elapsed))
        if self.active=='dit':
            self.engine.step()
            if number==self.start+self.count-1:self.finish()
        return result

    def wrap(self,target,name,label,decoder=False,writer=False,complete_file=False):
        original=getattr(target,name)
        book=self
        def wrapped(*args,**kwargs):
            if not book.enabled:return original(*args,**kwargs)
            if decoder and book.capture_vae and 'dit' in book.completed and 'vae_delivery' not in book.completed and book.engine is None:
                book.begin('vae_delivery')
            began=time.monotonic();failed=False
            try:
                with book.scope('h3/'+label) if book.engine else nullcontext():return original(*args,**kwargs)
            except BaseException:
                failed=True;raise
            finally:
                book.host.append(dict(operation=label,host_wall_sec=time.monotonic()-began,failed=failed))
                if writer:
                    frames=args[-1] if args else kwargs.get('decoded')
                    present=frames is not None and getattr(frames,'shape',(0,0,0))[2]>0
                    if present and book.first_decoded_frames_sec is None:book.first_decoded_frames_sec=time.monotonic()-book.started
                    if book.active=='vae_delivery':book.finish()
                if complete_file and not failed and book.first_complete_file_sec is None:
                    book.first_complete_file_sec=time.monotonic()-book.started
        setattr(target,name,wrapped)

    def summary(self):
        gpu=[]
        for index,start,end in self.events:
            gpu.append(dict(forward=index,current_cuda_stream_elapsed_ms=start.elapsed_time(end)))
        return dict(profile_trial=True,total_model_forwards=self.forward_index,profiled_windows=self.completed,
                    requested_dit_forward_start=self.start,requested_dit_forward_count=self.count,
                    model_calls=self.calls,cuda_stream_events=gpu,host_operations=self.host,
                    first_decoded_frames_available_sec=self.first_decoded_frames_sec,
                    first_complete_video_file_sec=self.first_complete_file_sec,
                    timing_origin='profile task start after shared model load; excludes queue/prep/Python import',
                    completed_video_file_stage='video-only container finalized before offline audio decode and AV mux',
                    first_playable_av_chunk_sec=None,
                    chunk_delivery_status='current sampler finalizes one whole video and muxes audio offline',
                    steady_unprofiled_counterpart_required=True,formal_streaming_profile_claim=False)


class ProfiledModel:
    def __init__(self,model,book):self.model=model;self.book=book
    def __call__(self,*args,**kwargs):
        return self.book.call_model(self.model,*args,**kwargs) if self.book.enabled else self.model(*args,**kwargs)
    def __getattr__(self,name):return getattr(self.model,name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--entry',choices=['fixed5','native_shot'],required=True)
    parser.add_argument('--start-forward',type=int,default=52);parser.add_argument('--forward-count',type=int,default=2)
    args=parser.parse_args();manifest=Path(args.manifest)
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    import torch
    import h3_shared_loader as loader
    rank=int(os.environ['RANK']);out=Path(os.environ['INFER_OUTPUT_DIR'])/'profile'
    engine=lambda:torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                         record_shapes=True,profile_memory=True,with_stack=False)
    book=ProfileBook(out,rank,args.start_forward,args.forward_count,engine,
                     scope_factory=torch.profiler.record_function,event_factory=lambda:torch.cuda.Event(enable_timing=True))
    load=loader.load_session
    def wrapped_load(*a,**kw):
        session=load(*a,**kw)
        book.wrap(session['audio_vae'],'decode','audio/offline_decode')
        return session
    loader.load_session=wrapped_load
    if args.entry=='native_shot':import h3_native_shot_batch_infer as entry
    else:import h3_batch_infer as entry
    render=entry.render
    def wrapped_render(session,task,*a,**kw):
        book.enabled=bool(task.get('profiling_trial',True));model=session['model']
        if book.enabled:
            book.started=time.monotonic();session['model']=ProfiledModel(model,book)
        try:return render(session,task,*a,**kw)
        finally:session['model']=model;book.enabled=False
    entry.render=wrapped_render
    for cls in [getattr(entry,'GeneratedVideoHistory',None),getattr(entry,'NativeHistory',None)]:
        if cls is None:continue
        for method in ['select','put','evict','commit','reset']:
            if hasattr(cls,method):book.wrap(cls,method,'history/'+method)
    book.wrap(entry.FixedChunkVideoDecoder,'push','video_vae/push',decoder=True)
    book.wrap(entry.FixedChunkVideoDecoder,'flush','video_vae/flush')
    book.wrap(entry.FrameWriter,'append','frames/encode_append',writer=True)
    book.wrap(entry.FrameWriter,'artifact','video/finalize_file',complete_file=True)
    packing='native_batch' if args.entry=='native_shot' else 'fixed_matched_batch'
    book.wrap(entry,packing,'history_and_condition_pack')
    sys.argv=[entry.__file__,'--manifest',str(manifest)]
    error=None
    try:entry.main()
    except BaseException:
        import traceback
        error=traceback.format_exc();raise
    finally:
        if book.engine is not None:book.finish()
        torch.cuda.synchronize()
        result=book.summary();result.update(error=error,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                           hardware_gpu=torch.cuda.get_device_name(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(out/f'rank{rank}_PROFILE_RESULT.json',result)


if __name__=='__main__':main()
