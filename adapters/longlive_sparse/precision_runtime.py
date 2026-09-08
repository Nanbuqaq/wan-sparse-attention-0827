"""Experimental causal whole-block precision runtime with a complete wire ledger.

This is an opt-in research extension to the existing pre-transfer branch. It
uses current post-RoPE Q representatives and committed compressed moments,
never full unselected candidate KV. Raw blocks and virtual prototypes are
different logical objects and are reported separately. CPU history is unbounded.
"""
from contextlib import AbstractContextManager
from dataclasses import asdict,dataclass
import hashlib
import inspect
import json
import time
import torch

from .ar_routing import build_route_plan
from .backends import BackendResult
from .moment_risk import OnlineMomentContext,score_raw_moment_risk
from .phase_prototypes import archive_rope0_key
from .prototype_wire import encode_prototype_frame,choose_whole_blocks,whole_block_token_indices
from .prototype_tail import PrototypeTail,execute_weighted_tail_sdpa
from .profiling import profiled


@dataclass(frozen=True)
class PrecisionRecipe:
    raw_density: float = .14
    query_samples: int = 1024
    groups_per_block: int = 4
    variance_codec: str = 'u8_scaled'
    admission: str = 'mass_key_variance'

    def __post_init__(self):
        if not 0<self.raw_density<1 or self.query_samples<1 or self.groups_per_block!=4:
            raise ValueError('invalid experimental precision recipe')
        if self.variance_codec not in ('bf16','u8_scaled') or self.admission not in ('mass_key_variance','mass_value','random'):
            raise ValueError('unsupported precision codec/admission')

    @classmethod
    def from_config(cls,config):
        return cls(raw_density=config.history_density,
            query_samples=config.method_params.get('precision_query_samples',1024),
            variance_codec=config.method_params.get('precision_variance_codec','u8_scaled'),
            admission=config.method_params.get('precision_admission','mass_key_variance'))


class WholeBlockPrecisionRuntime(AbstractContextManager):
    def __init__(self,pipeline,recipe=PrecisionRecipe()):
        self.pipeline=pipeline;self.recipe=recipe
        self.identity=hashlib.sha256(json.dumps(asdict(recipe),sort_keys=True).encode()).hexdigest()
        self.frames={};self.pending={};self.tails={};self.epoch=None;self.freqs=None;self.active=None
        self.hooks=[];self.records=[];self.index_records=[];self.last_summary_bytes=0
        self.ledger=dict(index_D2H_bytes=0,prototype_KV_H2D_bytes=0,routing_statistics_H2D_bytes=0,
            control_H2D_bytes=0,query_score_D2H_bytes=0,index_wall_s=0.,route_wall_s=0.,tail_prepare_wall_s=0.)
        self.ledger['candidate_KV_bytes_at_route_miss']=0

    def _check_epoch(self):
        epoch=self.pipeline.sparse_history_archive.epoch
        if epoch!=self.epoch:
            self.frames.clear();self.pending.clear();self.tails.clear();self.freqs=None;self.epoch=epoch

    def __enter__(self):
        archive=self.pipeline.sparse_history_archive
        if archive.config.rope_policy!='upstream_zero' or abs(archive.config.history_density-self.recipe.raw_density)>1e-9:
            raise ValueError('precision runtime requires upstream-zero RoPE and explicit matching raw density')
        if archive.config.refresh_policy!='per_chunk':raise ValueError('this experiment freezes one raw plan per chunk')
        if archive.config.method=='whole_block_precision_history' and self.recipe!=PrecisionRecipe.from_config(archive.config):
            raise ValueError('explicit precision method configuration differs from installed recipe')
        self.original_index=archive.index_frame
        for module in self.pipeline.sparse_history_modules:
            if getattr(module,'history_precision_runtime',None) is not None:raise ValueError('precision runtime already installed')
            signature=inspect.signature(module.forward)
            def before(owner,args,kwargs,signature=signature):
                self._check_epoch()
                bound=signature.bind_partial(*args,**kwargs).arguments
                self.active=(owner,bound)
                current=bound['freqs']
                if self.freqs is None:self.freqs=current.detach().clone()
                elif not torch.equal(self.freqs,current):raise RuntimeError('RoPE table changed during a committed precision epoch')
            self.hooks.append(module.register_forward_pre_hook(before,with_kwargs=True))
            module.history_precision_runtime=self;module.clear_selection_cache()
        archive.index_frame=self.index_frame
        return self

    @profiled('history/precision_index')
    def index_frame(self,layer_id,frame_id,key,value,**kwargs):
        self._check_epoch()
        if self.active is None or self.active[0].layer_id!=layer_id:raise RuntimeError('missing active commit owner')
        if key.device.type!='cuda':raise RuntimeError('index from already resident evicted KV, not onloaded old candidates')
        if key.dtype!=torch.bfloat16 or value.dtype!=torch.bfloat16:raise ValueError('this wire implementation requires BF16 source KV')
        indexed=self.original_index(layer_id,frame_id,key,value,**kwargs)
        start=time.perf_counter();archive=self.pipeline.sparse_history_archive
        roped=archive_rope0_key(key,spatial_height=archive.spatial_height,spatial_width=archive.spatial_width,
                                freqs=self.active[1]['freqs'])
        wire=encode_prototype_frame(roped,value,groups=4,variance_codec=self.recipe.variance_codec).cpu()
        torch.cuda.current_stream(key.device).synchronize()
        elapsed=time.perf_counter()-start
        self.frames[(int(layer_id),int(frame_id))]=(archive.frame_storage_version(layer_id,frame_id),wire)
        self.ledger['index_D2H_bytes']+=wire.bytes;self.ledger['index_wall_s']+=elapsed
        self.index_records.append(dict(layer=int(layer_id),frame=int(frame_id),wire_bytes=wire.bytes,index_wall_s=elapsed))
        return indexed

    @profiled('history/precision_route')
    def route(self,module,query,candidate_ids,*,exact_k_tokens,current_start):
        self._check_epoch();start=time.perf_counter();archive=self.pipeline.sparse_history_archive
        ids=tuple(int(x) for x in candidate_ids.detach().cpu())
        wires=[];versions=[]
        for frame in ids:
            version=archive.frame_storage_version(module.layer_id,frame)
            stored=self.frames.get((module.layer_id,frame))
            if stored is None or stored[0]!=version:raise RuntimeError('missing/stale committed prototype wire; no Dense fallback')
            versions.append(version);wires.append(stored[1])
        payload={}
        for field in ('key_mean','value_mean','key_variance','counts','variance_scale'):
            tensors=[getattr(w,field) for w in wires]
            if tensors[0] is None:continue
            cpu=torch.cat(tensors,dim=2)
            key='prototype_KV_H2D_bytes' if field in ('key_mean','value_mean','counts') else 'routing_statistics_H2D_bytes'
            self.ledger[key]+=cpu.numel()*cpu.element_size()
            payload[field]=cpu.to(query.device)
        variance=payload['key_variance'].float()
        if 'variance_scale' in payload:variance=variance*payload['variance_scale']
        n=min(self.recipe.query_samples,query.shape[1])
        representatives=torch.linspace(0,query.shape[1]-1,n,device=query.device).round().long()
        q=query.permute(0,2,1,3).index_select(2,representatives).float()
        context=OnlineMomentContext(q,q.square(),torch.ones(q.shape[:-1],device=q.device),
            payload['key_mean'],variance,payload['value_mean'],torch.zeros_like(payload['counts'],dtype=torch.float32),payload['counts'])
        if self.recipe.admission=='random':
            gen=torch.Generator().manual_seed(20260908+module.layer_id*1009+current_start)
            scores=torch.rand(query.shape[0],query.shape[2],payload['counts'].shape[-1]//4,generator=gen)
            summary_bytes=0
        else:
            scores=score_raw_moment_risk(context,candidate=self.recipe.admission).cpu()
            summary_bytes=scores.numel()*scores.element_size()
        self.last_summary_bytes=summary_bytes;self.ledger['query_score_D2H_bytes']+=summary_bytes
        frame_tokens=archive.spatial_height*archive.spatial_width;frames=len(ids);total=frame_tokens*frames
        self.ledger['candidate_KV_bytes_at_route_miss']+=query.shape[0]*query.shape[2]*total*query.shape[-1]*4
        widths=torch.tensor([min(64,frame_tokens-i) for _ in ids for i in range(0,frame_tokens,64)])
        blocks,raw_counts=choose_whole_blocks(scores,widths,token_budget=int(total*self.recipe.raw_density),
            byte_normalized=self.recipe.admission!='random')
        selections=[[[whole_block_token_indices(blocks[b][h],frame_tokens=frame_tokens,frames=frames)]
                     for h in range(query.shape[2])] for b in range(query.shape[0])]
        frame_ids=torch.tensor(ids).repeat_interleave(frame_tokens)[None,None].expand(query.shape[0],query.shape[2],-1)
        token_ids=torch.arange(frame_tokens).repeat(frames)[None,None].expand_as(frame_ids)
        plan=build_route_plan(method='whole_block_prototype_precision',routing_stage='pre-transfer',
            query_labels=torch.zeros(query.shape[0],query.shape[2],query.shape[1],dtype=torch.long),selections=selections,
            history_frame_ids=frame_ids,history_token_ids=token_ids,candidate_history_tokens=total,
            exact_k_tokens=exact_k_tokens,density=self.recipe.raw_density,
            metadata={'routing_identity':{'precision_recipe':asdict(self.recipe),'representation':'whole_raw_blocks_plus_complementary_Kmeans4_BF16_nodes'},
                      'teacher_used':False,'raw_whole_block_ids':blocks,'prototype_original_coverage':1.,
                      'virtual_nodes_are_not_exact_original_token_edges':True,'admission_cost':'known_raw_KV_bytes_only'})
        keep=torch.ones(payload['counts'].shape,dtype=torch.bool)
        for b in range(query.shape[0]):
            for h in range(query.shape[2]):
                for block in blocks[b][h]:keep[b,h,block*4:(block+1)*4]=False
        self.pending[module.layer_id]=(current_start,plan.digest(),archive.layer_storage_version(module.layer_id),payload,keep,raw_counts,total)
        self.ledger['route_wall_s']+=time.perf_counter()-start
        return plan

    @profiled('attention/precision_raw_plus_prototypes')
    def execute(self,module,query,exact_key,exact_value,history_key,history_value,plan,*,current_start):
        self._check_epoch();sha=plan.digest();archive=self.pipeline.sparse_history_archive
        cache_key=(archive.epoch,archive.layer_storage_version(module.layer_id),current_start,sha,str(query.dtype),str(query.device))
        hit=module.layer_id in self.tails and self.tails[module.layer_id][0]==cache_key
        if hit:tail,raw_counts=self.tails[module.layer_id][1:]
        else:
            begin=time.perf_counter()
            item=self.pending.pop(module.layer_id,None)
            if item is None or item[0]!=current_start or item[1]!=sha:raise RuntimeError('precision payload does not match the raw route')
            if item[2]!=archive.layer_storage_version(module.layer_id):raise RuntimeError('archive changed between precision selection and execution')
            _,_,_,payload,keep,raw_counts,total=item
            self.ledger['control_H2D_bytes']+=keep.numel()*keep.element_size()+raw_counts.numel()*raw_counts.element_size()
            mask=keep.to(query.device);raw_counts=raw_counts.to(query.device)
            counts=payload['counts'].float().masked_fill(~mask,0.)
            if not torch.equal(counts.sum(-1).long()+raw_counts,torch.full_like(raw_counts,total)):
                raise RuntimeError('raw/prototype coverage is missing or double-counted')
            tail=PrototypeTail(payload['key_mean'].permute(0,2,1,3),payload['value_mean'].permute(0,2,1,3),
                counts,64,archive.spatial_height*archive.spatial_width)
            self.tails[module.layer_id]=(cache_key,tail,raw_counts)
            self.ledger['tail_prepare_wall_s']+=time.perf_counter()-begin
        start=time.perf_counter()
        output=execute_weighted_tail_sdpa(query,exact_key,exact_value,history_key,history_value,tail,raw_valid_counts=raw_counts)
        torch.cuda.current_stream(query.device).synchronize();elapsed=time.perf_counter()-start
        logical=query.shape[1]*(query.shape[0]*query.shape[2]*exact_key.shape[1]+int(raw_counts.sum())+int((tail.counts>0).sum()))
        scheduled=query.shape[0]*query.shape[1]*query.shape[2]*(exact_key.shape[1]+history_key.shape[1]+tail.key.shape[1])
        self.records.append(dict(layer=module.layer_id,current_start=current_start,raw_route_sha256=sha,
            prototype_cache_hit=hit,raw_tokens=int(raw_counts.sum()),active_virtual_nodes=int((tail.counts>0).sum()),
            virtual_slots=tail.key.shape[1],attention_wall_s=elapsed))
        return BackendResult(output,'whole_precision_memory_efficient_sdpa',elapsed*1000,logical,scheduled,scheduled-logical,sha)

    def audit(self):
        return dict(recipe=asdict(self.recipe),recipe_sha256=self.identity,ledger=dict(self.ledger),records=self.records,
            index_records=self.index_records,extra_candidate_H2D_bytes=0,
            extra_metadata_H2D_bytes=sum(self.ledger[k] for k in ('prototype_KV_H2D_bytes','routing_statistics_H2D_bytes','control_H2D_bytes')),
            extra_index_D2H_bytes=self.ledger['index_D2H_bytes'],
            CPU_wire_bytes=sum(w.bytes for _,w in self.frames.values()),
            resident_tail_bytes=sum(t.bytes+counts.numel()*counts.element_size() for _,t,counts in self.tails.values()),
            complete_candidate_read_for_selection=False,future_video_access=False,
            unused_legacy_index_construction_still_charged=self.pipeline.sparse_history_archive.config.method!='whole_block_precision_history',
            represented_original_history_coverage=1.,virtual_edges_not_exact_original_edges=True,CPU_archive_bounded=False)

    def __exit__(self,exc_type,exc_value,tb):
        self.pipeline.sparse_history_archive.index_frame=self.original_index
        for hook in self.hooks:hook.remove()
        for module in self.pipeline.sparse_history_modules:
            del module.history_precision_runtime;module.clear_selection_cache()
        return False
