"""Explicit past-source geometry oracle, using unchanged causal source admission."""
import hashlib
import math
from pathlib import Path
import time
import torch
from .native_causal_block_memory import NativeCausalBlockMemory
from .history_cache import tensor_sha256


def fixed_mask_indices(foreground,*,source_tokens,budget,mode):
    ids=torch.as_tensor(foreground,dtype=torch.long,device='cpu').flatten()
    if ids.numel()!=ids.unique().numel() or not ids.numel() or int(ids.min())<0 or int(ids.max())>=source_tokens:
        raise ValueError('nonempty unique original foreground coordinates required')
    if not 0<budget<=source_tokens or mode not in ('foreground','background'):
        raise ValueError('explicit source budget and oracle mode required')
    mask=torch.zeros(source_tokens,dtype=torch.bool);mask[ids]=True
    background=torch.arange(source_tokens)[~mask]
    mandatory=ids if mode=='foreground' else torch.empty(0,dtype=torch.long)
    remaining=budget-mandatory.numel()
    if remaining<0 or remaining>background.numel():raise ValueError('oracle region does not fit this exact budget')
    if remaining:
        sites=((torch.arange(remaining,dtype=torch.float64)+.5)*background.numel()/remaining).floor().long()
        result=torch.cat([mandatory,background.index_select(0,sites)]).sort().values
    else:result=mandatory.sort().values
    if result.numel()!=budget or result.unique().numel()!=budget:raise RuntimeError('oracle fill lost exact unique budget')
    return result


class NativeOracleSourceMaskMemory(NativeCausalBlockMemory):
    def __init__(self,pipe,config,token_grid,*,mask_path,mask_mode='foreground'):
        if config.policy!='source_mask':raise ValueError('explicit source-mask method required')
        super().__init__(pipe,config,token_grid)
        path=Path(mask_path);self.mask_payload=torch.load(path,map_location='cpu',weights_only=True)
        if (self.mask_payload['source_end']-self.mask_payload['source_start']!=8
            or list(token_grid)!=self.mask_payload['token_grid']):raise ValueError('source mask geometry/version differs')
        self.mask_sha=hashlib.sha256(path.read_bytes()).hexdigest();self.mask_mode=mask_mode
        self.fixed_indices=fixed_mask_indices(self.mask_payload['indices'],source_tokens=8*self.frame_tokens,
            budget=math.floor(8*self.frame_tokens*config.fraction),mode=mask_mode)
        self.source_verified=False;self.oracle_used_layers=set()
        self.ledger.update(oracle_source_validation_D2H_bytes=0,oracle_source_validation_host_s=0.,oracle_index_prepare_host_s=0.)

    def after(self,owner,values,kwargs,result):
        super().after(owner,values,kwargs,result)
        frame=int(kwargs['current_start'])//self.frame_tokens
        if frame==self.mask_payload['source_start'] and self.scene.counts[frame]==5:
            value=kwargs['noisy_image_or_video'];started=time.perf_counter()
            actual=tensor_sha256(value)
            self.ledger['oracle_source_validation_D2H_bytes']+=value.numel()*value.element_size()
            self.ledger['oracle_source_validation_host_s']+=time.perf_counter()-started
            if actual!=self.mask_payload['source_latent_sha256']:raise RuntimeError('actual past source differs from oracle-mask source')
            self.source_verified=True

    def mask_source_indices(self,layer,heads,selected):
        if (not self.source_verified or self.active_bank['descriptor'].source_end!=self.mask_payload['source_end']
            or selected!=self.fixed_indices.numel()):raise RuntimeError('oracle mask cannot override a different/unchecked causal source')
        began=time.perf_counter();indices=self.fixed_indices[None].repeat(heads,1)
        self.oracle_used_layers.add(layer);self.ledger['oracle_index_prepare_host_s']+=time.perf_counter()-began
        return indices

    def audit(self):
        result=super().audit()
        result.update(method_variant='past_source_geometry_oracle',oracle=True,automatic_online_method=False,
            oracle_mask_sha256=self.mask_sha,oracle_mask_mode=self.mask_mode,actual_source_latents_verified=self.source_verified,
            source_mask_producer_scope=self.mask_payload.get('scope'),oracle_used_layers=sorted(self.oracle_used_layers),
            foreground_source_tokens=int(self.mask_payload['indices'].numel()),selected_source_tokens=int(self.fixed_indices.numel()),
            fixed_shared_selection_does_not_read_Q_or_teacher=True,exact_source_token_budget_with_final_group_trim=False,
            segmentation_generation_cost_external_not_in_this_model_timing=True,
            mask_CPU_tensor_bytes=self.mask_payload['indices'].numel()*self.mask_payload['indices'].element_size()+self.fixed_indices.numel()*self.fixed_indices.element_size())
        return result
