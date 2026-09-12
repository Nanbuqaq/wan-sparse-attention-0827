"""Single native Attention dispatcher for steady sparsity and full causal recall."""
import math
import time
import threading
import hashlib
import json

import torch

from .native_resident_history import (NativeResidentHistory, NativeResidentConfig,
    updated_frame_slots, window_slot_indices, contrast_scores, choose_whole_blocks)
from .native_summary_vectorized import summarize_frame_vectorized
from .native_causal_scene_memory import NativeCausalSceneMemory

METHODS=('w2_native','w2_steady_sparse','w2_full_recall','w2_steady_plus_recall')


def updated_owners(previous,info,current_start,frame_tokens,epoch,phase):
    owners=updated_frame_slots(previous,info,current_start,frame_tokens)
    begin,end=info['local_start_index']//frame_tokens,info['local_end_index']//frame_tokens
    first=current_start//frame_tokens
    owners[begin:end]=[('native',first+i,epoch,phase) for i in range(end-begin)]
    return owners


def classify_window(owners,physical,info,frame_tokens,effective_sink,global_sink,pin_start,pin_len):
    roles=[]
    actual_pin=pin_start-info.get('pinned_shift',0)
    for slot in physical:
        owner=owners[slot]
        if owner is None:raise RuntimeError('visible cache has no content owner')
        pos=slot*frame_tokens
        roles.append(dict(current=info['local_start_index']<=pos<info['local_end_index'],
            sink=pos<max(effective_sink,global_sink),pin=pin_len>0 and actual_pin<=pos<actual_pin+pin_len,
            recalled=owner[0]=='recalled'))
    return roles


class Wave2TemporalBudget(NativeResidentHistory):
    def __init__(self,pipe,method,*,fraction=.5,current_text,capture=False,
                 selector='mass_value',token_grid=None,preparation='old',route_audit=False,observer=False):
        if method not in METHODS[1:]:raise ValueError('native bypass must not install this adapter')
        super().__init__(pipe,NativeResidentConfig(policy='mass_value',fraction=fraction,
            reuse='none',summary_backend='vectorized'))
        self.method=method;self.steady=method in ('w2_steady_sparse','w2_steady_plus_recall')
        self.scene=NativeCausalSceneMemory(pipe) if method in ('w2_full_recall','w2_steady_plus_recall') else None
        self.current_text=current_text;self.phase_counts={};self.last_phase=None;self.cut_frame=0
        self.recall_frame=None;self.phase=0.;self.owners=[[None]*pipe.local_attn_size for _ in self.layers]
        self.metadata_cache={};self.storage_events=[];self.layout_records=[]
        self.metadata_builds=0;self.metadata_hits=0;self.invalidated_summaries=0
        self.metadata_GPU_peak_bytes=0;self.witness_lock=threading.Lock();self.clean_latent_hashes={}
        self.capture_enabled=capture;self.capture=None;self.feature_arrivals=[];self.feature_accesses=[];self.diagnostic_bytes=0;self.diagnostic_host_s=0.
        if selector not in ('mass_value','query_sum_batch4','query_balanced_batch4'):
            raise ValueError('unknown registered Wave2 selector')
        if selector!='mass_value' and (capture or token_grid is None or math.prod(token_grid)!=self.frame_tokens):
            raise ValueError('P3 requires explicit token grid and separate fixed-input diagnostics')
        self.selector=selector;self.token_grid=token_grid;self.head_metadata={}
        if preparation not in ('old','static_sort','deferred_stats','geometry_cache'):
            raise ValueError('unknown preparation ablation')
        self.preparation=preparation;self.defer_stats=preparation in ('deferred_stats','geometry_cache')
        self.route_audit=route_audit;self.route_hasher=hashlib.sha256();self.route_records=0
        self.stats_queue=[];self.stats_queue_bytes=0;self.stats_peak_bytes=0;self.stats_flush_host_s=0.
        self.stats_D2H_bytes=0;self.geometry_builds=0;self.geometry_hits=0
        if observer and (capture or not self.steady):raise ValueError('steady observer is separate from legacy capture')
        self.observer=observer;self.observer_calls=[];self.observer_arrivals=[];self.observer_accesses=[]
        self.observer_bytes=0;self.observer_host_s=0.

    def flush_statistics(self):
        if not self.stats_queue:return
        started=time.perf_counter()
        for device in {item[1].device for item in self.stats_queue}:
            items=[x for x in self.stats_queue if x[1].device==device]
            host=torch.stack([x[1] for x in items]).cpu().tolist()
            self.stats_D2H_bytes+=sum(x[1].numel()*x[1].element_size() for x in items)
            for (row,stats,mask,binding,heads,pairs,bytes_per_k),values in zip(items,host):
                if self.defer_stats:
                    selected=sum(values[:heads])/heads;actual=row['protected_union_tokens']+selected
                    row.update(selected_optional_per_head=values[:heads],coverage_min_per_head=values[heads:2*heads],
                        coverage_mean_per_head=values[2*heads:3*heads],physical_token_union=int(values[-1]),
                        selected_optional_tokens=selected,actual_K=actual,logical_pairs=pairs*actual,
                        GPU_gather_output_bytes=bytes_per_k*actual,statistics_finalized=True)
                if mask is not None:
                    raw=mask.cpu().numpy().tobytes();self.stats_D2H_bytes+=len(raw)
                    self.route_hasher.update(json.dumps(binding,separators=(',',':')).encode());self.route_hasher.update(raw)
                    self.route_records+=1
        self.stats_queue.clear();self.stats_queue_bytes=0
        self.stats_flush_host_s+=time.perf_counter()-started

    def observe_clean_latent(self,start,frames,digest):
        with self.witness_lock:self.clean_latent_hashes[start+frames]=digest

    def before(self,owner,values,kwargs):
        if self.pending:raise RuntimeError('uncommitted prior model call')
        self.active_start=int(kwargs['current_start']);frame=self.active_start//self.frame_tokens
        step=self.phase_counts.get(frame,0)
        if step>4:raise RuntimeError('unexpected denoising call count')
        self.phase_counts[frame]=step+1;self.clean=step==4;self.calls+=1
        if self.clean and not bool((kwargs['timestep']==0).all()):raise RuntimeError('clean commit timestep differs')
        self.phase=float(self.pipe._dit_model.rope_temporal_offset)
        if self.last_phase!=self.phase:self.cut_frame=frame;self.last_phase=self.phase
        if self.scene is not None:
            prior=len(self.scene.installations)
            self.scene.before(owner,values,kwargs,current_text=self.current_text(frame))
            if len(self.scene.installations)>prior:
                self.recall_frame=frame
                event=self.scene.installations[-1];installation=event['installation'];plan=installation['admission_plan']
                start,end=plan['destination_token_range']
                if start%self.frame_tokens or end%self.frame_tokens:raise RuntimeError('unaligned source binding')
                first,last=start//self.frame_tokens,end//self.frame_tokens
                for layer in range(len(self.layers)):
                    keys=[('recalled',f,plan['archive_version'],installation['KV_storage_version_sha256'],
                        plan['source_phase'],virtual,plan['bound_phase'])
                        for f,virtual in zip(plan['actual_source_frames'],plan['virtual_source_frames'])]
                    if len(keys)!=last-first:raise RuntimeError('source binding length differs')
                    self.owners[layer][first:last]=keys
                    valid=set(self.owners[layer]);bank=self.summaries[layer]
                    stale=[key for key in bank if key not in valid]
                    for key in stale:del bank[key]
                    self.invalidated_summaries+=len(stale);self.metadata_cache.pop(layer,None)
                self.storage_events.append(dict(model_call=self.calls,frame=frame,slot_range=[first,last],
                    source_frames=plan['actual_source_frames'],virtual_frames=plan['virtual_source_frames'],
                    source_phase=plan['source_phase'],bound_phase=plan['bound_phase'],archive_version=plan['archive_version'],
                    storage_version=installation['KV_storage_version_sha256']))

    def dispatch(self,layer,original,q,k,v,*,info,current_start,window_start,effective_sink,
                 pinned_start,pinned_len,prepend_sink,prepend_pinned,max_tokens,cache_end,global_sink_tokens):
        if current_start!=self.active_start or k.shape!=v.shape or k.dtype!=torch.bfloat16:
            raise RuntimeError('native dispatch identity or dtype differs')
        began=time.perf_counter();frame=current_start//self.frame_tokens
        owners=updated_owners(self.owners[layer],info,current_start,self.frame_tokens,self.calls,self.phase)
        physical=window_slot_indices(end=cache_end,start=window_start,effective_sink=effective_sink,
            pinned_start=pinned_start,pinned_len=pinned_len,prepend_sink=prepend_sink,
            prepend_pinned=prepend_pinned,max_tokens=max_tokens,frame_tokens=self.frame_tokens)
        if len(physical)*self.frame_tokens!=k.shape[1]:raise RuntimeError('window length differs from actual KV')
        roles=classify_window(owners,physical,info,self.frame_tokens,effective_sink,global_sink_tokens,pinned_start,pinned_len)
        protected_frames=[i for i,r in enumerate(roles) if any(r.values())]
        eligible=[(i,owners[slot]) for i,slot in enumerate(physical) if i not in protected_frames]
        candidate=len(eligible)*self.frame_tokens;protected=len(protected_frames)*self.frame_tokens
        state=('recalled_full' if frame==self.recall_frame else
               'native' if not self.steady or self.clean or frame==self.cut_frame or not eligible else 'steady_sparse')
        indices=None;selected=candidate;selection_s=0.;summary_s=0.;index_bytes=0
        head_chosen=None;head_metrics={};head_visible=None
        if state=='steady_sparse':
            key=(tuple(eligible),tuple(protected_frames),len(physical))
            cached=self.metadata_cache.get(layer)
            if cached is None or cached[0]!=key:
                token_blocks=[];counts=[]
                for position,_ in eligible:
                    for start in range(0,self.frame_tokens,64):
                        stop=min(start+64,self.frame_tokens)
                        token_blocks.append(range(position*self.frame_tokens+start,position*self.frame_tokens+stop));counts.append(stop-start)
                keep=[t for position in protected_frames for t in range(position*self.frame_tokens,(position+1)*self.frame_tokens)]
                cached=(key,token_blocks,counts,keep);self.metadata_cache[layer]=cached;self.metadata_builds+=1
            else:self.metadata_hits+=1
            _,token_blocks,counts,keep=cached[:4];bank=self.summaries[layer]
            if any(owner_key not in bank for _,owner_key in eligible):raise RuntimeError('missing/version-stale clean history summary')
            started=time.perf_counter()
            if len(cached)==4:
                summaries=[bank[owner_key] for _,owner_key in eligible]
                km=torch.cat([s[0] for s in summaries]);vm=torch.cat([s[1] for s in summaries]);count=torch.cat([s[2] for s in summaries])
                cached=(*cached,km,vm,count);self.metadata_cache[layer]=cached
                self.metadata_GPU_peak_bytes=max(self.metadata_GPU_peak_bytes,sum(t.numel()*t.element_size()
                    for record in self.metadata_cache.values() for t in record[4:]))
            else:km,vm,count=cached[4:]
            if self.selector=='mass_value':
                scores=contrast_scores(q[0],km,vm,count,'mass_value',samples=32).cpu().tolist()
                chosen,selected,budget=choose_whole_blocks(scores,counts,self.config.fraction)
                coordinates=sorted(keep+[t for group in chosen for t in token_blocks[group]])
                indices=torch.tensor(coordinates,device=k.device,dtype=torch.long);index_bytes=indices.numel()*indices.element_size()
            else:
                from .query_balanced_value import stratified_sites,normalized_values,select_batched,select_static_once
                meta=self.head_metadata.get(layer)
                geometry_key=(len(physical),tuple(protected_frames),tuple(pos for pos,_ in eligible),tuple(counts),q.shape[2],self.token_grid)
                map_key=geometry_key if self.preparation=='geometry_cache' else key
                if meta is None or meta[0]!=map_key:
                    mapping=torch.full((k.shape[1],),-2,dtype=torch.long);mapping[keep]=-1
                    for group,tokens in enumerate(token_blocks):mapping[list(tokens)]=group
                    if (mapping==-2).any():raise RuntimeError('incomplete head route mapping')
                    sites=stratified_sites(8,*self.token_grid)
                    index_bytes=mapping.numel()*8+sites.numel()*8
                    meta=(map_key,mapping.to(k.device),sites.to(q.device));self.head_metadata[layer]=meta;self.geometry_builds+=1
                else:self.geometry_hits+=1
                _,head_mapping,sites=meta
                a=normalized_values(q[0,sites],km,vm,count);budget=math.floor(candidate*self.config.fraction)
                if self.selector=='query_sum_batch4' and self.preparation!='old':
                    head_chosen,coverage,used=select_static_once(a,counts,budget)
                else:
                    head_chosen,coverage,used=select_batched(a,counts,budget,
                        balanced=self.selector=='query_balanced_batch4')
                # One explicit diagnostic readback is charged, not hidden in a
                # GPU-only selection claim. Original head-specific KV follows.
                stats=torch.cat([used.float(),coverage.amin(1),coverage.mean(1)]);heads=q.shape[2]
                head_metrics=dict(selector_a_bytes=a.numel()*a.element_size(),selector_stats_D2H_bytes=0,
                    head_metadata_GPU_bytes=sum(t.numel()*t.element_size() for m in self.head_metadata.values() for t in m[1:]))
                if not self.defer_stats:
                    host=stats.cpu().tolist();selected=sum(host[:heads])/heads
                    head_metrics.update(selected_optional_per_head=host[:heads],coverage_min_per_head=host[heads:2*heads],
                        coverage_mean_per_head=host[2*heads:],selector_stats_D2H_bytes=len(host)*4)
            selection_s=time.perf_counter()-started
        prepare_s=time.perf_counter()-began
        if head_chosen is not None:
            from .query_balanced_value import execute_per_head
            output,head_visible=execute_per_head(q,k,v,head_chosen,head_mapping,protected+budget)
            audit_started=time.perf_counter()
            union=head_visible.any(0).sum()
            if self.defer_stats:
                stats=torch.cat([stats,union.float()[None]])
                head_metrics['stats_enqueue_host_s']=time.perf_counter()-audit_started
            else:
                head_metrics['physical_token_union']=int(union.cpu())
                head_metrics['post_execution_audit_host_s']=time.perf_counter()-audit_started
                head_metrics['selector_stats_D2H_bytes']+=8
                stats=torch.cat([stats,union.float()[None]])
        else:
            output=original(q,k,v) if indices is None else original(q,k.index_select(1,indices),v.index_select(1,indices))
        if self.observer and layer==14:
            observed=time.perf_counter()
            if state=='steady_sparse':
                access=dict(frame=frame,call=self.calls,eligible=eligible,protected_positions=protected_frames,
                    selected_group_mask=head_chosen.cpu() if head_chosen is not None else None,
                    selected_shared_groups=chosen if head_chosen is None else None)
                if access['selected_group_mask'] is not None:self.observer_bytes+=access['selected_group_mask'].numel()
                self.observer_accesses.append(access)
            else:
                self.observer_accesses.append(dict(frame=frame,call=self.calls,state=state,
                    full_visible_owners=[owners[x] for x in physical]))
            if frame in (24,88) and self.phase_counts[frame]==1 and state=='steady_sparse':
                record=dict(frame=frame,layer=layer,q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),
                    output=output.detach().cpu(),key_mean=km.cpu(),value_mean=vm.cpu(),counts=count.cpu(),
                    token_blocks=[list(x) for x in token_blocks],protected=keep,eligible=eligible,
                    selected_group_mask=head_chosen.cpu() if head_chosen is not None else None,
                    selected_indices=indices.cpu() if indices is not None else None,
                    frame_tokens=self.frame_tokens,token_grid=self.token_grid,physical=physical,
                    owners=[owners[x] for x in physical],roles=roles)
                self.observer_bytes+=sum(x.numel()*x.element_size() for x in record.values() if isinstance(x,torch.Tensor))
                self.observer_calls.append(record)
            self.observer_host_s+=time.perf_counter()-observed
        if self.capture_enabled and layer==14:
            accessed=[]
            spans=([range(i*self.frame_tokens,(i+1)*self.frame_tokens) for i in range(len(physical))]
                if indices is None else [range(i*self.frame_tokens,(i+1)*self.frame_tokens) for i in protected_frames]+[token_blocks[i] for i in chosen])
            for span in spans:
                position=span.start//self.frame_tokens;key=owners[physical[position]]
                if key[0]=='native' and key[1]<frame:
                    start=key[1]*self.frame_tokens+span.start%self.frame_tokens
                    accessed.append([start,start+len(span)])
            self.feature_accesses.append(dict(frame=frame,clean=self.clean,state=state,ranges=accessed))
        if self.capture_enabled and self.capture is None and frame==24 and layer==14 and state=='steady_sparse':
            started=time.perf_counter()
            self.capture=dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),output=output.detach().cpu(),
                selected_indices=indices.cpu(),key_mean=km.cpu(),value_mean=vm.cpu(),counts=count.cpu(),
                token_blocks=[list(x) for x in token_blocks],protected=keep,eligible=eligible,
                frame_tokens=self.frame_tokens,config=self.config.__dict__,frame=frame,layer=layer)
            self.diagnostic_bytes+=sum(t.numel()*t.element_size() for t in self.capture.values() if isinstance(t,torch.Tensor))
            self.diagnostic_host_s+=time.perf_counter()-started
        additions={}
        if self.clean and self.steady:
            started=time.perf_counter();new_k,new_v=info['new_k'][0],info['new_v'][0]
            first=info['local_start_index']//self.frame_tokens
            for offset in range(0,new_k.shape[0],self.frame_tokens):
                key=owners[first+offset//self.frame_tokens]
                additions[key]=summarize_frame_vectorized(new_k[offset:offset+self.frame_tokens],new_v[offset:offset+self.frame_tokens])
            summary_s=time.perf_counter()-started;self.summary_build_host_s+=summary_s
            if self.observer and layer==14:
                observed=time.perf_counter()
                if new_k.shape[0]%16:raise RuntimeError('Block16 observer requires aligned clean tokens')
                item=dict(frame=frame,rope_phase=self.phase,frame_tokens=self.frame_tokens,
                    key_mean=new_k.reshape(-1,16,*new_k.shape[1:]).float().mean(1).cpu(),
                    value_mean=new_v.reshape(-1,16,*new_v.shape[1:]).float().mean(1).cpu(),
                    counts=torch.full((new_k.shape[0]//16,),16,device=new_k.device,dtype=torch.long).cpu(),
                    owners=list(additions),group_atom_tokens=16,
                    feature_scope='all independent heads retained; post-RoPE Block16 original clean KV means')
                self.observer_bytes+=sum(x.numel()*x.element_size() for x in item.values() if isinstance(x,torch.Tensor))
                self.observer_arrivals.append(item);self.observer_host_s+=time.perf_counter()-observed
            if self.capture_enabled and layer==14:
                started=time.perf_counter();feature=new_k.float().mean(1).cpu()
                self.feature_arrivals.append(dict(frame=frame,rope_phase=self.phase,key_features=feature,source='actual new clean KV, head mean',
                    accessed_frames=[key[1] for _,key in eligible if key[0]=='native']))
                self.diagnostic_bytes+=feature.numel()*feature.element_size();self.diagnostic_host_s+=time.perf_counter()-started
        self.pending[layer]=(owners,additions)
        if self.diagnostic_bytes>512*1024**2:raise RuntimeError('Wave2 diagnostic tensor budget exceeded')
        if self.observer_bytes>1024**3:raise RuntimeError('steady observer exceeds registered1GiB CPU tensor budget')
        actual=(protected+selected if head_chosen is not None else k.shape[1] if indices is None else indices.numel());pairs=q.shape[1]*q.shape[2]
        self.rows.append(dict(call=self.calls,layer=layer,current_frame=frame,clean_commit=self.clean,state=state,
            current_tokens=sum(r['current'] for r in roles)*self.frame_tokens,
            sink_tokens=sum(r['sink'] for r in roles)*self.frame_tokens,
            pin_tokens=sum(r['pin'] for r in roles)*self.frame_tokens,
            recalled_tokens=sum(r['recalled'] for r in roles)*self.frame_tokens,
            protected_union_tokens=protected,optional_tokens=candidate,selected_optional_tokens=selected,
            native_K=k.shape[1],actual_K=actual,logical_pairs=pairs*actual,full_native_pairs=pairs*k.shape[1],
            selection_host_s=selection_s,prepare_host_s=prepare_s,summary_host_s=summary_s,index_H2D_bytes=index_bytes,
            GPU_gather_output_bytes=0 if indices is None and head_chosen is None else 2*actual*k.shape[2]*k.shape[3]*k.element_size(),
            backend='native_FA2_varlen_per_head' if head_chosen is not None else 'native_FA2',
            selector=self.selector,**head_metrics,route_reused=False,summary_version_keys_checked=True))
        if head_chosen is not None and (self.defer_stats or self.route_audit):
            mask=head_chosen.detach() if self.route_audit else None
            binding=(self.calls,layer,frame,eligible,protected_frames)
            self.stats_queue.append((self.rows[-1],stats.detach(),mask,binding,q.shape[2],pairs,2*k.shape[2]*k.shape[3]*k.element_size()))
            self.stats_queue_bytes+=stats.numel()*stats.element_size()+(mask.numel()*mask.element_size() if mask is not None else 0)
            self.stats_peak_bytes=max(self.stats_peak_bytes,self.stats_queue_bytes)
            if self.stats_queue_bytes>2*1024**2:raise RuntimeError('bounded statistics queue exceeded2MiB')
        if self.clean:
            self.layout_records.append(dict(layer=layer,frame=frame,physical_slots=physical,
                owners=[owners[x] for x in physical],roles=roles))
        return output

    def after(self,owner,values,kwargs,result):
        for layer,(owners,additions) in self.pending.items():
            self.owners[layer]=owners;bank=self.summaries[layer];bank.update(additions);valid=set(owners)
            for key in list(bank):
                if key not in valid:del bank[key]
        self.pending.clear();self.active_start=None
        size=sum(t.numel()*t.element_size() for bank in self.summaries for value in bank.values() for t in value)
        self.summary_peak_bytes=max(self.summary_peak_bytes,size)
        if self.scene is not None:self.scene.after(owner,values,kwargs,result)
        if self.clean:self.flush_statistics()

    def audit(self):
        self.flush_statistics()
        with self.witness_lock:witness=dict(self.clean_latent_hashes)
        return dict(method=self.method,selector=self.selector,preparation=self.preparation,
            route_audit_sha256=self.route_hasher.hexdigest() if self.route_audit else None,route_audit_records=self.route_records,
            deferred_statistics_peak_bytes=self.stats_peak_bytes,deferred_statistics_flush_host_s=self.stats_flush_host_s,
            deferred_statistics_D2H_bytes=self.stats_D2H_bytes,geometry_builds=self.geometry_builds,geometry_hits=self.geometry_hits,
            steady_observer=self.observer,observer_D2H_tensor_bytes=self.observer_bytes,observer_host_s=self.observer_host_s,
            rows=self.rows,layouts=self.layout_records,storage_events=self.storage_events,
            scene=self.scene.audit() if self.scene else None,summary_GPU_peak_bytes=self.summary_peak_bytes,
            summary_build_host_s=self.summary_build_host_s,metadata_builds=self.metadata_builds,metadata_hits=self.metadata_hits,
            invalidated_summaries=self.invalidated_summaries,optional_fraction=self.config.fraction,
            metadata_GPU_peak_bytes=self.metadata_GPU_peak_bytes,own_clean_latent_hashes=witness,
            source_hashes=[dict(archive_version=a['archive_version'],source_end=a['source_end'],
                own_clean_latent_sha256=witness.get(a['source_end'])) for a in self.scene.archives] if self.scene else [],
            source_witness_not_selector_input=True,diagnostic_D2H_bytes=self.diagnostic_bytes,
            diagnostic_host_s=self.diagnostic_host_s,
            clean_commit='full_native',cut_first_chunk='full_native',recalled_chunk='full_native',
            one_attention_dispatch_per_layer=True,denoise_route_reuse=False,raw_KV_unchanged=True,
            source_binding_changes_are_versioned=True,no_new_archive_without_recall=self.scene is None)
