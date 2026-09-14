"""Same-route reference: pack mandatory KV once across spatial query groups.

Only the current invocation's immutable input KV is shared. No current noisy KV,
query, output, or probability is reused across denoising calls. The two exact
softmax partitions are combined with their independently computed log normalizers.
"""
import torch


def execute_shared_mandatory_routes(q, k, v, plan, timing=None):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    frames, meta = plan['frame_ids'], plan['meta']
    groups, heads, count = frames.shape
    mandatory = meta['protected']
    common_count = mandatory.numel()
    residual_count = count - common_count
    if groups != 4 or common_count < 1 or residual_count < 1:
        raise ValueError('reference requires four groups and both nonempty partitions')
    if q.shape[0] != 1 or q.dtype != torch.bfloat16 or k.shape != v.shape:
        raise ValueError('one BF16 raw-KV window required')
    if timing is not None: timing[0].record()
    # Fixed output sizes avoid data-dependent nonzero/masked_select host waits.
    shared_mask = (frames[..., None] == mandatory).any(-1)
    sentinel = k.shape[1] // meta['offsets'].numel()
    residual = torch.where(shared_mask, sentinel, frames).sort(-1).values[..., :residual_count]
    offsets = meta['offsets']
    common_tokens = (mandatory[:, None] * offsets.numel() + offsets).flatten()
    ck = k.index_select(1, common_tokens)
    cv = v.index_select(1, common_tokens)
    indices = (residual[..., None] * offsets.numel() + offsets).flatten(-2)
    h = torch.arange(heads, device=q.device)[None, :, None]
    rk = k[0].permute(1,0,2)[h,indices].reshape(-1,1,q.shape[-1])
    rv = v[0].permute(1,0,2)[h,indices].reshape(-1,1,q.shape[-1])
    qids = meta['query_ids']
    rq = q[0,qids].permute(0,2,1,3).contiguous().reshape(-1,1,q.shape[-1])
    cuk = torch.arange(groups*heads+1,device=q.device,dtype=torch.int32)*(residual_count*offsets.numel())
    if timing is not None: timing[1].record()
    co, clse, cp = flash_attn_func(q,ck,cv,dropout_p=0.,causal=False,return_attn_probs=True)
    ro, rlse, rp = flash_attn_varlen_func(rq,rk,rv,meta['cuq'],cuk,qids.shape[1],
        residual_count*offsets.numel(),dropout_p=0.,causal=False,return_attn_probs=True)
    if cp.numel() or rp.numel():
        raise RuntimeError('unexpected probability matrix materialization')
    if timing is not None: timing[2].record()
    ro = ro.reshape(groups,heads,qids.shape[1],q.shape[-1]).permute(0,2,1,3).reshape(-1,heads,q.shape[-1])
    ro = ro.index_select(0,meta['inverse_queries'])[None]
    rlse = rlse.reshape(groups,heads,qids.shape[1]).permute(0,2,1).reshape(-1,heads)
    rlse = rlse.index_select(0,meta['inverse_queries'])[None]
    weight = torch.sigmoid(rlse - clse.transpose(1,2)).unsqueeze(-1)
    result = (co.float() + weight * (ro.float()-co.float())).to(q.dtype)
    if timing is not None: timing[3].record()
    return result
