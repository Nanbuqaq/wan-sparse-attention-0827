"""Archive-time past-K groups, reusing the gated source selection/install path."""
from dataclasses import replace
import hashlib
from pathlib import Path
import time
import torch
from .native_causal_block_memory import NativeCausalBlockMemory
from .native_balanced_key_groups import balanced_key_groups


def pack_uniform_partition(groups,tokens):
    if not groups or len({len(g) for g in groups})!=1 or not 0<len(groups[0])<=64:
        raise ValueError('qualified balanced source groups must have one width at most64')
    if sorted(i for g in groups for i in g)!=list(range(tokens)):
        raise ValueError('partition must cover each original token once')
    return torch.tensor(groups,dtype=torch.int32),torch.full((len(groups),),len(groups[0]),dtype=torch.int64)


class NativeKeySourceMemory(NativeCausalBlockMemory):
    def __init__(self,pipe,config,token_grid):
        if config.grouping not in ('key_frame','key_bank','flat_key_matched') or token_grid not in ((22,40),(8,16)):
            raise ValueError('qualified native key partition geometry required')
        if config.normalization!='source_only' or config.refresh!='first_only':
            raise ValueError('first key-group slice holds normalization and route lifetime fixed')
        super().__init__(pipe,replace(config,grouping='flat64'),token_grid)
        self.config=config;self.groups=[];self.group_counts=torch.empty(0,dtype=torch.int64)
        self.ledger.update(partition_build_host_nested_s=0.,partition_index_D2H_bytes=0,
            partition_counts_H2D_bytes=0,partition_lookup_host_nested_s=0.,CPU_partition_tensor_peak_bytes=0)

    def _archive_with_groups(self,frame):
        self._sample_memory('before_key_archive_'+str(frame));self.scene._archive_last_scene(frame)
        started=time.perf_counter();bank=self.scene.banks[-1];means=[];parts=[];extra=0;mean_bytes=0
        n=8*self.frame_tokens
        for cache in self.pipe.kv_cache_pos:
            end=int(cache['local_end_index']);source_k=cache['k'][0,end-n:end]
            began=time.perf_counter()
            if self.config.grouping=='key_bank':groups=balanced_key_groups(source_k)
            elif self.config.grouping=='key_frame':
                groups=[[i+f*self.frame_tokens for i in g] for f in range(8)
                    for g in balanced_key_groups(source_k[f*self.frame_tokens:(f+1)*self.frame_tokens])]
            else:
                width=55 if self.frame_tokens==880 else 64
                groups=[list(range(i,i+width)) for i in range(0,n,width)]
            self.ledger['partition_build_host_nested_s']+=time.perf_counter()-began
            if self.config.grouping!='flat_key_matched':self.ledger['partition_index_D2H_bytes']+=n*8
            indices,counts=pack_uniform_partition(groups,n);parts.append((indices,counts))
            extra+=indices.numel()*indices.element_size()+counts.numel()*counts.element_size()
            index_pad=torch.zeros(len(groups),64,dtype=torch.long);mask=torch.zeros_like(index_pad,dtype=torch.bool)
            width=indices.shape[1];index_pad[:,:width]=indices.long();mask[:,:width]=True
            self.ledger['group_index_H2D_bytes']+=sum(t.numel()*t.element_size() for t in (index_pad,mask,counts))
            gi,gm,gc=index_pad.to(source_k.device),mask.to(source_k.device),counts.to(source_k.device)
            values=[]
            for name in ('k','v'):
                source=cache[name][0,end-n:end]
                scratch=source.index_select(0,gi.flatten()).float().reshape(len(groups),64,*source.shape[1:])
                scratch.masked_fill_(~gm[:,:,None,None],0.)
                mean=(scratch.sum(1)/gc[:,None,None]).cpu().clone()
                values.append(mean);size=mean.numel()*mean.element_size();extra+=size;mean_bytes+=size
            means.append(tuple(values))
        bank['group_means']=means;bank['key_partitions']=parts;bank['owned_bytes']+=extra
        self._sample_memory('key_archive_before_eviction_'+str(frame))
        while sum(b['owned_bytes'] for b in self.scene.banks)>self.scene.budget:
            self.scene.banks.pop(0);self.scene.ledger['evicted_archives']+=1
        self.scene.ledger['CPU_archive_peak_tensor_bytes']=max(self.scene.ledger['CPU_archive_peak_tensor_bytes'],sum(b['owned_bytes'] for b in self.scene.banks))
        index_bytes=sum(t.numel()*t.element_size() for b in self.scene.banks for pair in b.get('key_partitions',[]) for t in pair)
        self.ledger['CPU_partition_tensor_peak_bytes']=max(self.ledger['CPU_partition_tensor_peak_bytes'],index_bytes)
        self.ledger['group_summary_D2H_bytes']+=mean_bytes
        self.ledger['group_prepare_host_s']+=time.perf_counter()-started
        self._sample_memory('after_key_archive_'+str(frame))

    def scoring_groups(self,layer):
        began=time.perf_counter();groups=self.active_bank['key_partitions'][layer][0].tolist()
        self.ledger['partition_lookup_host_nested_s']+=time.perf_counter()-began
        return groups

    def scoring_counts(self,layer,device):
        counts=self.active_bank['key_partitions'][layer][1]
        self.ledger['partition_counts_H2D_bytes']+=counts.numel()*counts.element_size()
        return counts.to(device)

    def audit(self):
        result=super().audit()
        result.update(partitions_created_from_past_GPU_K_at_scene_closure_before_recall_rebinding=True,
            no_query_or_teacher_used_for_partition=True,
            partition_index_CPU_storage_included_in_archive_budget=True,
            partition_build_scope_nested_in_group_prepare=True,
            partition_lookup_scope_nested_in_source_scoring=True,
            key_grouping_not_claimed_as_full_ClusterVLM_reproduction=True)
        return result

    def export_routes(self,path):
        result=super().export_routes(path)
        target=Path(path).with_name('source_key_partitions.pt');records=[]
        for bank in self.scene.banks:
            descriptor=bank['descriptor']
            for layer,(indices,counts) in enumerate(bank['key_partitions']):
                records.append(dict(archive_version=descriptor.archive_version,source_end=descriptor.source_end,
                    source_phase=descriptor.source_phase,layer=layer,indices=indices,counts=counts))
        with target.open('xb') as handle:
            torch.save(dict(schema='native_source_key_partitions_v1',config=self.config.__dict__,
                max_group_tokens=64,bisection_iterations=6,retained_banks_only=True,records=records),handle)
        with target.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        result['partition_snapshot']=dict(path=str(target),sha256=digest,records=len(records))
        return result
