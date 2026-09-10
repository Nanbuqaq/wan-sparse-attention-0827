"""Query-aware source blocks, with unchanged coarse causal scene selection.

Qualified explicit-cut/local32/global8/pin8 path. Source data is installed once
per layer when first-return Q exists, into existing native pinned storage. Only
valid source slots participate until the native next-chunk eviction. Full source
is an exact control; partial source is a distinct algorithm, not pure layout.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import resource
import time
import torch

from .native_resident_history import NativeResidentHistory,NativeResidentConfig
from .native_causal_scene_memory import NativeCausalSceneMemory
from .native_scene_admission import choose_scene,has_revisit_cue
from .native_retimed_memory import binding_coordinates
from .native_temporal_rephase import rephase_temporal_keys


@dataclass(frozen=True)
class CausalBlockConfig:
    policy: str = 'full'
    fraction: float = 1.
    grouping: str = 'flat64'
    head_policy: str = 'shared'
    query_samples: int = 32
    random_seed: int = 20260910
    normalization: str = 'source_only'
    refresh: str = 'first_only'
    query_reduction: str = 'mean'
    def __post_init__(self):
        if self.policy not in ('full','random','mass_value','contrast_value','source_mask'):
            raise ValueError('unknown source-block policy')
        if not 0 < self.fraction <= 1 or (self.policy=='full' and self.fraction!=1):
            raise ValueError('full control needs fraction1; partial budget must be explicit')
        if self.grouping not in ('flat64','spatial8','flat_matched','key_frame','key_bank','flat_key_matched','spacetime2x4','flat_tube_matched') or self.head_policy not in ('shared','per_head'):
            raise ValueError('unknown grouping/head policy')
        if self.grouping in ('key_frame','key_bank','flat_key_matched') and (self.policy not in ('mass_value','contrast_value') or self.fraction>=1):
            raise ValueError('key-group slice requires an explicit partial value-scoring policy')
        if self.normalization not in ('source_only','joint_context') or (self.normalization=='joint_context' and self.policy not in ('mass_value','contrast_value')):
            raise ValueError('joint context applies only to declared value-scoring policies')
        if self.refresh not in ('first_only','phase2') or (self.refresh=='phase2' and (self.policy not in ('mass_value','contrast_value') or self.fraction>=1)):
            raise ValueError('mid-denoising refresh requires a partial value-scoring source policy')
        if self.policy=='source_mask' and (self.grouping!='flat64' or self.head_policy!='shared' or self.normalization!='source_only' or self.refresh!='first_only'):
            raise ValueError('source-mask oracle uses shared fixed coordinates and its original lifetime')
        if self.query_reduction not in ('mean','normalized_peak') or (self.query_reduction!='mean'
            and (self.policy!='mass_value' or self.normalization!='source_only' or self.refresh!='first_only'
                 or self.grouping!='spatial8' or self.head_policy!='per_head')):
            raise ValueError('normalized query peak requires the isolated spatial8 per-head mass-value slice')
        if self.grouping in ('spacetime2x4','flat_tube_matched') and (self.policy!='mass_value'
            or self.normalization!='source_only' or self.refresh!='first_only' or self.head_policy!='per_head'):
            raise ValueError('time grouping uses the isolated per-head mass-value slice')


def groups_for_source(height,width,frames=8,kind='flat64'):
    groups=[];tokens=height*width
    if kind in ('spacetime2x4','flat_tube_matched'):
        if frames!=8 or height%2 or width%4:
            raise ValueError('time groups require eight source frames and 2x4 spatial alignment')
        if kind=='flat_tube_matched':
            return [list(range(a,a+64)) for a in range(0,frames*tokens,64)]
        return [[frame*tokens+yy*width+xx for frame in range(frames)
                 for yy in range(y,y+2) for xx in range(x,x+4)]
                for y in range(0,height,2) for x in range(0,width,4)]
    for frame in range(frames):
        base=frame*tokens
        if kind=='flat64':
            groups.extend([list(range(base+a,base+min(a+64,tokens))) for a in range(0,tokens,64)])
        elif kind=='spatial8':
            for y in range(0,height,8):
                for x in range(0,width,8):
                    groups.append([base+yy*width+xx for yy in range(y,min(y+8,height)) for xx in range(x,min(x+8,width))])
        elif kind=='flat_matched':
            begin=base
            for group in groups_for_source(height,width,frames=1,kind='spatial8'):
                groups.append(list(range(begin,begin+len(group))));begin+=len(group)
        else:raise ValueError('unknown source partition')
    if sorted(t for g in groups for t in g)!=list(range(frames*tokens)):
        raise ValueError('source partition must cover each original token exactly once')
    return groups


def gather_source_heads(source,indices):
    """CPU original [1,S,H,D] -> canonical per-head [1,n,H,D]."""
    if source.shape[0]!=1 or indices.shape[0]!=source.shape[2]:raise ValueError('head geometry differs')
    if indices.numel() and (int(indices.min())<0 or int(indices.max())>=source.shape[1]):raise ValueError('source index out of range')
    heads=torch.arange(source.shape[2],device=indices.device)[:,None]
    return source[0].permute(1,0,2)[heads,indices].permute(1,0,2)[None].contiguous()


def reduce_query_scores(score,reduction):
    if reduction=='mean':return score.mean(1)
    if reduction=='normalized_peak':
        relative=score/score.sum(-1,keepdim=True).clamp_min(torch.finfo(score.dtype).tiny)
        return relative.amax(1)
    raise ValueError('unknown query score reduction')


def source_head_scores(q,km,vm,counts,policy,samples=32,reduction='mean'):
    sites=torch.linspace(0,q.shape[0]-1,min(q.shape[0],samples),device=q.device).round().long()
    qq=q.index_select(0,sites).float()
    logits=torch.einsum('qhd,ghd->hqg',qq,km.float())/math.sqrt(q.shape[-1])
    p=(logits+counts.float().log()[None,None,:]).softmax(-1)
    values=vm.float().permute(1,0,2)
    if policy=='mass_value':score=p*values.norm(dim=-1)[:,None,:]
    elif policy=='contrast_value':
        mixture=torch.einsum('hqg,hgd->hqd',p,values)
        score=p*(values[:,None,:,:]-mixture[:,:,None,:]).norm(dim=-1)
    else:raise ValueError('not a source scoring policy')
    return reduce_query_scores(score,reduction)


def exact_source_indices(groups,scores,budget):
    order=sorted(range(len(groups)),key=lambda i:(-scores[i],i))
    return sorted([t for i in order for t in groups[i]][:budget])


class NativeCausalBlockMemory(NativeResidentHistory):
    def __init__(self,pipe,config,token_grid):
        super().__init__(pipe,NativeResidentConfig())
        if pipe.global_sink_size!=8 or pipe.sink_size!=8 or pipe.local_attn_size!=32:
            raise ValueError('first source-block slice requires native global8/pin8/local32')
        if any(not getattr(b.self_attn,'_research_inplace_cache',False) for b in self.layers):
            raise ValueError('source-block installation requires the gated in-place native cache')
        if pipe._dit_model.t_scale!=1 or pipe._dit_model.rope_method!='linear' or pipe._dit_model.original_seq_len is not None:
            raise ValueError('qualified native linear scale1 temporal binding required')
        self.config=config;self.grid=token_grid;self.scene=NativeCausalSceneMemory(pipe)
        self.groups=groups_for_source(*token_grid,kind=config.grouping)
        self.group_counts=torch.tensor([len(g) for g in self.groups],dtype=torch.int64)
        self.active_bank=None;self.target_frame=None;self.masks={};self.served=set();self.routes_saved=[]
        self.active_phase=0;self.refresh_events=[]
        self.group_gpu=None;self.current_text=None
        # KV allocation is lazy; the generator has already been placed by the runner.
        self.memory_samples=[];self.device=next(pipe._dit_model.parameters()).device
        self.ledger=dict(group_prepare_host_s=0.,group_index_H2D_bytes=0,group_summary_D2H_bytes=0,
            group_summary_H2D_bytes=0,source_KV_H2D_bytes=0,CPU_selected_pack_read_write_logical_bytes=0,
            CPU_pack_host_s=0.,source_install_host_s=0.,source_rebind_host_s=0.,
            source_score_host_including_readiness_s=0.,source_ranking_CPU_s=0.,route_mask_H2D_bytes=0,
            score_result_D2H_bytes=0,group_index_GPU_resident_bytes=0,
            kept_summary_prepare_host_nested_s=0.,kept_summary_input_logical_GPU_bytes=0,
            kept_summary_tensor_peak_bytes=0,kept_summary_counts_H2D_bytes=0)

    def _sample_memory(self,event):
        # The native runner releases its KV dictionaries before VAE/final audit.
        # Retain only the device identity, never a cache tensor to keep it alive.
        device=self.device
        self.memory_samples.append(dict(event=event,process_lifetime_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            GPU_allocated_bytes=torch.cuda.memory_allocated(device),GPU_reserved_bytes=torch.cuda.memory_reserved(device),
            CPU_archive_tensor_bytes=sum(b['owned_bytes'] for b in self.scene.banks)))

    def attach(self,current_text):
        self.current_text=current_text
        super().attach()

    def scoring_groups(self,layer):
        return self.groups

    def scoring_counts(self,layer,device):
        return self.group_gpu[2]

    def _archive_with_groups(self,frame):
        self._sample_memory('before_archive_'+str(frame))
        self.scene._archive_last_scene(frame)
        if self.config.policy in ('full','random','source_mask'):
            self._sample_memory('after_raw_archive_'+str(frame));return
        started=time.perf_counter();bank=self.scene.banks[-1]
        if self.group_gpu is None:
            idx=torch.zeros(len(self.groups),64,dtype=torch.long);mask=torch.zeros_like(idx,dtype=torch.bool)
            for i,g in enumerate(self.groups):idx[i,:len(g)]=torch.tensor(g);mask[i,:len(g)]=True
            device=self.pipe.kv_cache_pos[0]['k'].device
            self.group_gpu=(idx.to(device),mask.to(device),self.group_counts.to(device))
            self.ledger['group_index_H2D_bytes']+=sum(t.numel()*t.element_size() for t in (idx,mask,self.group_counts))
            self.ledger['group_index_GPU_resident_bytes']=self.ledger['group_index_H2D_bytes']
        idx,mask,counts=self.group_gpu;means=[];extra=0
        for cache in self.pipe.kv_cache_pos:
            end=int(cache['local_end_index']);row=[]
            for name in ('k','v'):
                source=cache[name][0,end-8*self.frame_tokens:end]
                scratch=source.index_select(0,idx.flatten()).float().reshape(len(self.groups),64,*source.shape[1:])
                scratch.masked_fill_(~mask[:,:,None,None],0.)
                mean=(scratch.sum(1)/counts[:,None,None]).cpu().clone()
                row.append(mean);extra+=mean.numel()*mean.element_size()
            means.append(tuple(row))
        bank['group_means']=means;bank['owned_bytes']+=extra
        self._sample_memory('group_archive_before_eviction_'+str(frame))
        while sum(b['owned_bytes'] for b in self.scene.banks)>self.scene.budget:
            self.scene.banks.pop(0);self.scene.ledger['evicted_archives']+=1
        total=sum(b['owned_bytes'] for b in self.scene.banks)
        self.scene.ledger['CPU_archive_peak_tensor_bytes']=max(total,self.scene.ledger['CPU_archive_peak_tensor_bytes'])
        self.ledger['group_summary_D2H_bytes']+=extra
        self.ledger['group_prepare_host_s']+=time.perf_counter()-started
        self._sample_memory('after_group_archive_'+str(frame))

    def before(self,owner,values,kwargs):
        super().before(owner,values,kwargs)
        frame=self.active_start//self.frame_tokens
        phase=self.scene.counts.get(frame,0);self.scene.counts[frame]=phase+1
        self.active_phase=phase
        if phase:
            if self.config.refresh=='phase2' and phase==2 and self.active_bank is not None and frame==self.target_frame:
                self.masks={};self.refresh_events.append(dict(at_latent=frame,phase=phase))
            return
        if self.target_frame is not None and frame!=self.target_frame:
            if frame!=self.target_frame+8:raise RuntimeError('unexpected source lifetime transition')
            for cache in self.pipe.kv_cache_pos:
                if int(cache['pinned_start'])!=int(cache['local_end_index'])-8*self.frame_tokens:
                    raise RuntimeError('first return did not become the next native anchor')
            self.target_frame=None;self.active_bank=None;self.masks={}
        rope_phase=float(self.pipe._dit_model.rope_temporal_offset)
        cut=self.scene.last_commit is not None and rope_phase!=self.scene.last_commit['phase']
        if cut:self._archive_with_groups(frame)
        if rope_phase in self.served:return
        text=self.current_text(frame)
        prototype=self.scene._prototype(kwargs['conditional_dict']) if has_revisit_cue(text) else None
        started=time.perf_counter()
        decision=choose_scene(text,prototype,[b['descriptor'] for b in self.scene.banks],frame,**self.scene.policy)
        self.scene.ledger['selector_CPU_wall_s']+=time.perf_counter()-started
        self.scene.decisions.append(dict(at_latent=frame,**decision))
        if decision['selected_version'] is None:return
        if not cut:raise RuntimeError('first source-block slice requires an explicit new revisit shot')
        self.active_bank=next(b for b in self.scene.banks if b['descriptor'].archive_version==decision['selected_version'])
        self.target_frame=frame;self.masks={};source=self.active_bank['descriptor']
        self.binding=binding_coordinates(source_frame=source.source_end-8,target_frame=frame,frames=8,
            source_phase=source.source_phase,current_phase=rope_phase,policy='recent_virtual')

    def after(self,owner,values,kwargs,result):
        self.scene.after(owner,values,kwargs,result)
        if self.active_bank is not None:
            if len(self.masks)!=len(self.layers):raise RuntimeError('source installation did not reach every layer')
            phase=float(self.pipe._dit_model.rope_temporal_offset)
            if phase not in self.served:
                self.served.add(phase)
                self.scene.installations.append(dict(at_latent=self.target_frame,current_phase=phase,
                    source_archive_version=self.active_bank['descriptor'].archive_version,binding=self.binding,
                    source_selected_per_head=math.floor(8*self.frame_tokens*self.config.fraction),
                    lifetime='first_return_chunk_then_native_pin_and_eviction'))
        self.active_start=None

    def dispatch(self,layer,original,q,k,v,*,info,current_start,window_start,effective_sink,
                 pinned_start,pinned_len,prepend_sink,prepend_pinned,max_tokens,cache_end,global_sink_tokens):
        if current_start!=self.active_start:raise RuntimeError('dispatch outside active generator call')
        active=self.active_bank is not None and current_start//self.frame_tokens==self.target_frame
        mask=None;selected=0;source_tokens=8*self.frame_tokens
        if active:
            if (window_start or prepend_sink or prepend_pinned or pinned_len!=source_tokens
                or pinned_start-info.get('pinned_shift',0)!=global_sink_tokens
                or global_sink_tokens!=source_tokens or k.shape[1]!=32*self.frame_tokens):
                raise RuntimeError('unregistered source slot/window geometry')
            cache=self.pipe.kv_cache_pos[layer];destination=global_sink_tokens
            if (k.untyped_storage().data_ptr()!=cache['k'].untyped_storage().data_ptr()
                or v.untyped_storage().data_ptr()!=cache['v'].untyped_storage().data_ptr()):
                raise RuntimeError('window must alias the gated native cache')
            selected=math.floor(source_tokens*self.config.fraction)
            if layer not in self.masks:
                source_k,source_v=self.active_bank['kv'][layer]
                if self.config.policy=='full':
                    ids=None
                    packed_k,packed_v=source_k,source_v
                elif self.config.policy=='source_mask':
                    ids=self.mask_source_indices(layer,q.shape[2],selected)
                    packed=time.perf_counter()
                    packed_k=gather_source_heads(source_k,ids);packed_v=gather_source_heads(source_v,ids)
                    size=packed_k.numel()*packed_k.element_size()+packed_v.numel()*packed_v.element_size()
                    self.ledger['CPU_selected_pack_read_write_logical_bytes']+=2*size
                    self.ledger['CPU_pack_host_s']+=time.perf_counter()-packed
                else:
                    score_start=time.perf_counter()
                    groups=self.scoring_groups(layer)
                    if self.config.policy=='random':
                        seed=self.config.random_seed+layer+1000*self.target_frame
                        rng=random.Random(seed)
                        scores=[[rng.random() for _ in groups] for _ in range(q.shape[2])]
                    else:
                        km,vm=self.active_bank['group_means'][layer]
                        self.ledger['group_summary_H2D_bytes']+=sum(t.numel()*t.element_size() for t in (km,vm))
                        km,vm=km.to(q.device),vm.to(q.device)
                        km=rephase_temporal_keys(km,self.binding['temporal_delta'])
                        score_counts=self.scoring_counts(layer,q.device)
                        if self.config.normalization=='source_only':
                            score_tensor=source_head_scores(q[0],km,vm,score_counts,self.config.policy,self.config.query_samples,self.config.query_reduction)
                        else:
                            from .native_source_context_proxy import kept_context_means,source_context_scores
                            began=time.perf_counter()
                            kk,kv,kc=kept_context_means(k,v,source_start=destination,source_tokens=source_tokens,frame_tokens=self.frame_tokens)
                            self.ledger['kept_summary_prepare_host_nested_s']+=time.perf_counter()-began
                            self.ledger['kept_summary_input_logical_GPU_bytes']+=(k.shape[1]-source_tokens)*k.shape[0]*k.shape[2]*k.shape[3]*(k.element_size()+v.element_size())
                            self.ledger['kept_summary_tensor_peak_bytes']=max(self.ledger['kept_summary_tensor_peak_bytes'],sum(t.numel()*t.element_size() for t in (kk,kv,kc)))
                            self.ledger['kept_summary_counts_H2D_bytes']+=kc.numel()*kc.element_size()
                            score_tensor=source_context_scores(q[0],km,vm,score_counts,kk,kv,kc,self.config.policy,self.config.query_samples)
                        self.ledger['score_result_D2H_bytes']+=score_tensor.numel()*score_tensor.element_size()
                        scores=score_tensor.cpu().tolist()
                    self.ledger['source_score_host_including_readiness_s']+=time.perf_counter()-score_start
                    rank_start=time.perf_counter()
                    if self.config.head_policy=='shared':
                        scores=[[sum(s[i] for s in scores)/len(scores) for i in range(len(groups))]]*q.shape[2]
                    ids=torch.tensor([exact_source_indices(groups,s,selected) for s in scores],dtype=torch.long)
                    self.ledger['source_ranking_CPU_s']+=time.perf_counter()-rank_start
                    packed=time.perf_counter()
                    packed_k=gather_source_heads(source_k,ids);packed_v=gather_source_heads(source_v,ids)
                    size=packed_k.numel()*packed_k.element_size()+packed_v.numel()*packed_v.element_size()
                    self.ledger['CPU_selected_pack_read_write_logical_bytes']+=2*size
                    self.ledger['CPU_pack_host_s']+=time.perf_counter()-packed
                started=time.perf_counter()
                cache['k'][:,destination:destination+selected].copy_(packed_k)
                cache['v'][:,destination:destination+selected].copy_(packed_v)
                self.ledger['source_KV_H2D_bytes']+=sum(t.numel()*t.element_size() for t in (packed_k,packed_v))
                self.ledger['source_install_host_s']+=time.perf_counter()-started
                started=time.perf_counter();key=cache['k'][:,destination:destination+selected]
                key.copy_(rephase_temporal_keys(key,self.binding['temporal_delta']))
                self.ledger['source_rebind_host_s']+=time.perf_counter()-started
                keep=list(range(destination+selected))+list(range(destination+source_tokens,k.shape[1]))
                mask=None if selected==source_tokens else torch.tensor(keep,device=q.device,dtype=torch.long)
                if mask is not None:self.ledger['route_mask_H2D_bytes']+=mask.numel()*mask.element_size()
                self.masks[layer]=mask
                self.routes_saved.append(dict(frame=self.target_frame,layer=layer,phase=self.active_phase,
                    archive_version=self.active_bank['descriptor'].archive_version,
                    source_start=self.active_bank['descriptor'].source_end-8,
                    source_indices=ids.int() if ids is not None else None,full_source_canonical=ids is None,
                    destination_start=destination,binding=dict(self.binding)))
            mask=self.masks[layer]
        if mask is None:output=original(q,k,v);actual=k.shape[1]
        else:output=original(q,k.index_select(1,mask),v.index_select(1,mask));actual=mask.numel()
        self.rows.append(dict(call=self.calls,layer=layer,frame=current_start//self.frame_tokens,phase=self.active_phase,
            active_source=active,selected_source_tokens_per_head=selected,
            source_candidate_tokens=source_tokens if active else 0,
            actual_K=actual,native_K=k.shape[1],logical_pairs=q.shape[1]*q.shape[2]*actual,
            native_pairs=q.shape[1]*q.shape[2]*k.shape[1],backend='native_FA2'))
        return output

    def export_routes(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        torch.save(dict(schema='native_causal_block_routes_v1',config=self.config.__dict__,records=self.routes_saved),path)
        with path.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
        return dict(path=str(path),sha256=sha,records=len(self.routes_saved))

    def audit(self):
        self._sample_memory('audit')
        return dict(schema='native_causal_block_memory_v2',config=self.config.__dict__,rows=self.rows,
            scene_selection=self.scene.audit(),ledger=self.ledger,
            refresh_events=self.refresh_events,
            memory_samples=self.memory_samples,
            memory_sample_scope='RSS is process lifetime peak including loading; GPU readings are sampled whole-process allocator bytes, not stage-local peaks',
            kept_summary_prepare_scope_nested_in_source_score=True,
            kept_summary_input_bytes_are_logical_not_HBM_counter=True,
            saved_route_CPU_tensor_bytes=sum(r['source_indices'].numel()*r['source_indices'].element_size()
                for r in self.routes_saved if r['source_indices'] is not None),
            candidate_domain='one_coarse_selected_source_bank_not_all_CPU_history',
            exact_source_token_budget_with_final_group_trim=True,
            K_temporally_rebound_like_baseline=True,V_and_spatial_K_unchanged=True,
            original_source_tokens_executed_not_prototypes=True,
            unused_source_slots_excluded_not_zero_filled=True,
            first_return_Q_selection_frozen_through_clean=self.config.refresh=='first_only',
            input_forward_sha256=self.source_sha256,derived_forward_sha256=self.derived_sha256)
