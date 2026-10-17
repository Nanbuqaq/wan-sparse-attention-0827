"""Content interventions keep every retained state row and its position intact.

Only projected state content is masked. Retained zero rows still participate in
attention; this is not a token-removal ablation or a new memory architecture.
"""
import torch

POLICIES=('full','original_content','recent_content','wrong_past')


def content_mask(ids,original_ids,recent_ids,policy,rows_per_frame,device='cpu'):
    if policy not in POLICIES:raise ValueError('Unknown history-content intervention')
    if ids!=sorted(set(ids)) or len(set(original_ids)&set(recent_ids)):
        raise ValueError('History IDs and role groups must be ordered and disjoint')
    if set(ids)!=set(original_ids)|set(recent_ids):raise ValueError('Role groups must cover exact retained history')
    if type(rows_per_frame) is not int or rows_per_frame<=0:raise ValueError('Positive patch rows per history latent required')
    if policy in ['full','wrong_past']:return None
    allowed=set(original_ids if policy=='original_content' else recent_ids)
    return torch.tensor([i in allowed for i in ids],dtype=torch.bool,device=device).repeat_interleave(rows_per_frame)[None,:,None]


def mask_projected(output,mask):
    if mask is None:return output
    if output.ndim!=3 or tuple(mask.shape)!=(output.shape[0],output.shape[1],1) or mask.dtype!=torch.bool:
        raise ValueError('Projected state mask changes batch/row geometry')
    return output.masked_fill(~mask,0)


def choose_past(current,donor,policy):
    if policy not in POLICIES:raise ValueError('Unknown history policy')
    if current is None:raise ValueError('Content diagnostic requires actual generated history')
    if policy!='wrong_past':return current
    if donor is None or donor.shape!=current.shape or donor.dtype!=current.dtype:
        raise ValueError('Wrong past must exactly match retained video geometry and precision')
    return donor
