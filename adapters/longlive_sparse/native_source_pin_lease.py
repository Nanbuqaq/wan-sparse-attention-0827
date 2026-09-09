"""Registered source-pin lifetime diagnostic, not a general state validator.

After first-return clean commit, keep the actual recalled source pin instead
of the newly generated return pin. No K/V is copied/rotated here. Sampled
witnesses read real resident K/V; they are diagnostic traffic, not a selector.
"""
import hashlib
import time

import torch


class NativeSourcePinLease:
    def __init__(self,pipe,memory):
        if pipe.local_attn_size!=32 or pipe.global_sink_size!=8 or pipe.num_frame_per_block!=8:
            raise ValueError('source lease is qualified only for local32/global8/block8')
        self.pipe,self.memory=pipe,memory;self.active=None;self.reference=None;self.indices=None;self.handle=None
        self.events=[];self.checks=[];self.checked_frames=set()
        self.ledger=dict(extra_raw_KV_H2D_bytes=0,metadata_GPU_write_bytes=0,metadata_readback_bytes=0,
            witness_index_H2D_bytes=0,witness_KV_D2H_bytes=0,policy_wall_s=0.,witness_wall_including_readiness_s=0.)

    def _sample(self):
        begin=time.perf_counter();caches=self.pipe.kv_cache_pos;sample=[]
        layers=sorted({0,(len(caches)-1)//2,len(caches)-1})
        for layer in layers:
            for name in ('k','v'):
                value=caches[layer][name].index_select(1,self.indices).detach().cpu()
                if caches[layer][name].device.type=='cuda':self.ledger['witness_KV_D2H_bytes']+=value.numel()*value.element_size()
                sample.append((layer,name,value.clone()))
        self.ledger['witness_wall_including_readiness_s']+=time.perf_counter()-begin
        return sample

    def pin(self,original,caches,current_frames):
        last=self.memory.last_commit
        entry=self.memory.installations[-1] if self.memory.installations else None
        phase=float(self.pipe._dit_model.rope_temporal_offset)
        applies=(entry is not None and last is not None and last['end']==entry['at_latent']+8
                 and phase==entry['current_phase'])
        if not applies:return original(caches,current_frames)
        if caches is not self.pipe.kv_cache_pos or current_frames!=8:raise ValueError('unqualified cache side/chunk')
        plan=entry['installation']['admission_plan'];start,end=plan['destination_token_range']
        if start!=8*self.pipe.frame_seq_length or end-start!=8*self.pipe.frame_seq_length:
            raise ValueError('only the actual source adjacent to global sink can be leased')
        began=time.perf_counter()
        before=(int(caches[0]['pinned_start']),int(caches[0]['pinned_len']))
        self.ledger['metadata_readback_bytes']+=16
        if before!=(start,end-start):raise RuntimeError('source pin moved before the registered intervention')
        original(caches,current_frames)
        requested=(int(caches[0]['pinned_start']),int(caches[0]['pinned_len']))
        self.ledger['metadata_readback_bytes']+=16
        for cache in caches:cache['pinned_start'].fill_(start);cache['pinned_len'].fill_(end-start)
        self.ledger['metadata_GPU_write_bytes']+=16*len(caches)
        self.ledger['policy_wall_s']+=time.perf_counter()-began
        points=sorted({0,self.pipe.frame_seq_length//3,2*self.pipe.frame_seq_length//3,self.pipe.frame_seq_length-1})
        indices=[start+f*self.pipe.frame_seq_length+i for f in range(8) for i in points]
        self.indices=torch.tensor(indices,device=caches[0]['k'].device,dtype=torch.long)
        if self.indices.device.type=='cuda':self.ledger['witness_index_H2D_bytes']+=self.indices.numel()*self.indices.element_size()
        self.active=dict(phase=phase,start=start,length=end-start,installed_at=entry['at_latent'])
        self.reference=self._sample();self.checked_frames=set()
        digest=hashlib.sha256()
        for layer,name,value in self.reference:
            digest.update(f'{layer}/{name}'.encode());digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        self.events.append(dict(completed_latent=last['end'],phase=phase,source_frames=plan['source_frames'],
            requested_native_pin=list(requested),effective_source_pin=[start,end-start],
            parent_admission_plan_sha256=entry['installation']['admission_plan_sha256'],
            sampled_resident_KV_sha256=digest.hexdigest(),logical_selection_changed=True,
            K_or_V_rewritten_by_lease=False,full_KV_equality_claimed=False))

    def before(self,owner,values,kwargs):
        if self.active is None:return
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        if float(self.pipe._dit_model.rope_temporal_offset)!=self.active['phase']:
            self.active=None
            return
        if frame in self.checked_frames:return
        self.checked_frames.add(frame)
        first=self.pipe.kv_cache_pos[0]
        actual=(int(first['pinned_start']),int(first['pinned_len']));self.ledger['metadata_readback_bytes']+=16
        if actual!=(self.active['start'],self.active['length']):raise RuntimeError('active source lease metadata changed')
        sample=self._sample()
        if any(a[:2]!=b[:2] or not torch.equal(a[2],b[2]) for a,b in zip(sample,self.reference)):
            raise RuntimeError('sampled original source K/V changed while leased')
        self.checks.append(dict(query_start_latent=frame,sampled_resident_KV_exact=True,pin=list(actual)))

    def attach(self):self.handle=self.pipe.generator.register_forward_pre_hook(self.before,with_kwargs=True)

    def detach(self):
        if self.handle is not None:self.handle.remove();self.handle=None

    def audit(self):
        return dict(policy='keep_recalled_source_pin_after_first_return_clean',events=self.events,checks=self.checks,ledger=self.ledger,
            samples='up to three layers, all heads, four fixed positions per source frame',
            sampled_witness_not_full_KV_audit=True,diagnostic_traffic_must_not_be_hidden=True,
            not_general_state_completion_detection=True,uncut_intent_change_not_qualified=True,
            future_native_scene_pins_not_overridden_without_a_new_matching_installation=True)
