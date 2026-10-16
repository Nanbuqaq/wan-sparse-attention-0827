"""Matched conditional flow with spatial past patches and absolute native AV time.

This is teacher forcing. It does not implement generated-history training, a KV
cache, long-term retention, or the published TaoMate training recipe.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_activation
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep,patchify_video_latents
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb


class SpatialHistoryProjection(nn.Linear):
    def __init__(self, native_projection):
        super().__init__(native_projection.in_features,native_projection.out_features,bias=native_projection.bias is not None)
        with torch.no_grad():
            self.weight.copy_(native_projection.weight.detach().float().cpu())
            if self.bias is not None:self.bias.copy_(native_projection.bias.detach().float().cpu())

    def forward(self, value):
        # Keep an FP32 master adapter for AdamW, but avoid materializing the
        # long [history_tokens, hidden_size] projection result in FP32.  H3's
        # transformer consumes BF16 state rows, so BF16 matmul is the exact
        # downstream dtype and saves a large transient on long windows.
        compute_dtype = torch.bfloat16 if value.is_cuda else self.weight.dtype
        weight = self.weight.to(compute_dtype)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(value.to(compute_dtype), weight, bias)


class HistoryReadAttention:
    """Exact prefix/current attention using fused Flash SDPA, without S-by-S logits.

    Prefix queries read only prefix keys. Current queries read the entire
    document. Query row slicing is a compute tile, never a data subchunk.
    """
    _attention_backend = None
    _parallel_config = None

    def __init__(self, query_chunk_size=8192, query_checkpoint=True):
        self.prefix_rows = None
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.query_checkpoint = bool(query_checkpoint)

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None, is_causal=False):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if attention_mask is not None or is_causal:
            raise ValueError("Only explicit prefix/current block attention is supported")
        p = self.prefix_rows
        if p is None or not 0 < p < hidden_states.shape[1]:
            raise ValueError("Missing spatial-history prefix length")
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        if rotary_emb is not None:
            key = _apply_rotary_emb(key, *rotary_emb)
        key = key.transpose(1, 2)
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)

        def query_tile(x, k, v, cos, sin):
            q = attn.norm_q(attn.to_q(x).unflatten(-1, (attn.heads, -1)))
            if rotary_emb is not None:
                q = _apply_rotary_emb(q, cos, sin)
            q = q.transpose(1, 2)
            if x.is_cuda:
                # Fail explicitly if flash is unavailable; never silently use math attention.
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            else:
                result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            result = result.transpose(1, 2).flatten(2, 3)
            return attn.to_out[1](attn.to_out[0](result))

        outputs = []
        prefix_keys = hidden_states.shape[1] if getattr(self, 'prefix_reads_current', False) else p
        for lo, hi, visible in ((0, p, prefix_keys), (p, hidden_states.shape[1], hidden_states.shape[1])):
            for start in range(lo, hi, self.query_chunk_size):
                end = min(hi, start + self.query_chunk_size)
                cos, sin = (rotary_emb[0][start:end], rotary_emb[1][start:end]) if rotary_emb is not None else (None, None)
                args = (hidden_states[:, start:end], key[:, :, :visible], value[:, :, :visible], cos, sin)
                if torch.is_grad_enabled() and self.query_checkpoint:
                    outputs.append(checkpoint_activation(query_tile, *args, use_reentrant=False))
                else:
                    outputs.append(query_tile(*args))
        return torch.cat(outputs, dim=1)


def install_history_attention(transformer, query_chunk_size=8192, query_checkpoint=True):
    processors = []
    for block in transformer.transformer_blocks:
        processor = HistoryReadAttention(query_chunk_size, query_checkpoint)
        block.attn.set_processor(processor)
        processors.append(processor)
    return processors


def matched_spatial_batch(video,audio,prompt,tags,audio_valid_mask,generator,device,past_latents=None,history_frames=4,history_latents=20,target_latents=5,target_start_latents=None,target_end_latents=None,history_start_latents=None,history_end_latents=None,target_audio_start_latents=None,target_audio_end_latents=None):
    if video.ndim!=4 or audio.ndim!=3 or audio.shape[:2]!=(2,32):raise ValueError("Require video[C,T,H,W], audio[2,32,A]")
    c,vf,h,w=video.shape;af=audio.shape[-1]
    if c!=24 or h%2 or w%2:raise ValueError("Invalid native video geometry")
    if target_start_latents is None:
        if history_latents < 2 or (history_latents-2)%5:
            raise ValueError("history_latents must be 2+5*n for native H3 geometry")
        target_start_latents = history_latents
    if target_end_latents is None:
        target_end_latents = min(vf, target_start_latents + target_latents)
    if history_end_latents is None:
        history_end_latents = target_start_latents if past_latents is None else past_latents
    if history_start_latents is None:
        history_start_latents = 0
    if not 0 <= history_start_latents <= history_end_latents <= target_start_latents or target_end_latents <= target_start_latents or target_end_latents > vf:
        raise ValueError("Invalid variable history/target boundaries")
    if target_start_latents and target_start_latents < 2 or (target_start_latents >= 2 and (target_start_latents-2)%5):
        raise ValueError("Target start boundary must be 0 or 2+5*n")
    target_end = target_end_latents
    # Native standalone prefix 5 pixels/2 latents; subsequent groups 17 pixels/5 latents.
    cut_pixel_frame=0 if target_start_latents == 0 else 5+17*((target_start_latents-2)//5)
    audio_cut=round(cut_pixel_frame*40/24) if target_audio_start_latents is None else int(target_audio_start_latents)
    if not 0<=audio_cut<af:raise ValueError("Audio target is empty at native global boundary")
    source_video=video.to(device,torch.float32);source_audio=audio.to(device,torch.float32)
    # Audio follows the native pixel-frame boundary.  Five latent frames map to
    # one 17-pixel-frame group, so using target latent count directly would shift
    # the audio boundary and can produce an empty or misaligned target.
    target_pixel_end=0 if target_end == 0 else 5+17*((target_end-2)//5)
    target_audio_end=round(target_pixel_end*40/24) if target_audio_end_latents is None else int(target_audio_end_latents)
    target_audio_end=min(af, target_audio_end)
    if target_audio_end <= audio_cut:
        raise ValueError("Current audio target is empty after source-end clamp")
    target_video=source_video[:,target_start_latents:target_end][None]
    target_audio=source_audio[:,:,audio_cut:target_audio_end].permute(0,2,1).reshape(-1,32)
    epsilon_v=torch.randn(target_video.shape,generator=generator,device=device)
    epsilon_a=torch.randn(target_audio.shape,generator=generator,device=device)
    u=torch.rand((),generator=generator,device=device).clamp(1e-4,1-1e-4)
    sigma_v=12*u/(1+11*u);sigma_a=3*u/(1+2*u)
    t_v,t_a=1-sigma_v,1-sigma_a
    video_rows=patchify_video_latents(t_v*target_video+sigma_v*epsilon_v,(1,2,2))
    audio_rows=t_a*target_audio+sigma_a*epsilon_a
    target_v=patchify_video_latents(target_video-epsilon_v,(1,2,2))
    target_a=target_audio-epsilon_a
    raw=MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags.cpu(),vf,h,w,af,(1,2,2),2,2,0)
    pos,_,vi,ai,ti,_,_=raw
    rows_per_frame=(h//2)*(w//2)
    available_history = history_end_latents - history_start_latents
    if history_frames > 0:
        history_ids=torch.linspace(history_start_latents,history_end_latents-1,min(history_frames,available_history)).round().long().unique()
    else:
        history_ids=torch.arange(history_start_latents,history_end_latents).long()
    history_row_ids=(history_ids[:,None]*rows_per_frame+torch.arange(rows_per_frame)[None]).reshape(-1)
    if history_row_ids.numel():
        local_history_ids=history_ids-history_start_latents
        local_row_ids=(local_history_ids[:,None]*rows_per_frame+torch.arange(rows_per_frame)[None]).reshape(-1)
        clean_history_rows=patchify_video_latents(source_video[:,history_start_latents:history_end_latents][None],(1,2,2)).index_select(0,local_row_ids.to(device))
    else:
        clean_history_rows=torch.empty((0, c*4),device=device,dtype=source_video.dtype)
    # Some zero-frame reshape paths retain the spatial product in the trailing
    # dimension even though they contain no rows.  Normalize the empty tensor
    # explicitly so the history projection always receives [N, 96].
    if clean_history_rows.numel() == 0:
        clean_history_rows = torch.empty((0, c*4), device=device, dtype=source_video.dtype)
    else:
        clean_history_rows = clean_history_rows.reshape(-1, c*4)
    past_pos=pos.index_select(0,vi.index_select(0,history_row_ids))
    current_video_pos=pos.index_select(0,vi[target_start_latents*rows_per_frame:target_end*rows_per_frame])
    current_audio_ids=torch.cat([ai[ch*af+audio_cut:ch*af+target_audio_end] for ch in range(2)])
    current_audio_pos=pos.index_select(0,current_audio_ids)
    text_len=ti.numel();history_count=len(history_row_ids);prefix=text_len+history_count
    position_ids=torch.cat([pos[ti],past_pos,current_audio_pos,current_video_pos]).to(device)
    text_indices=torch.arange(text_len,device=device)
    state_indices=torch.arange(text_len,prefix,device=device)
    audio_indices=torch.arange(prefix,prefix+len(audio_rows),device=device)
    video_indices=torch.arange(prefix+len(audio_rows),len(position_ids),device=device)
    token_tags=torch.cat([tags.cpu(),torch.zeros(history_count,dtype=torch.long),torch.full((len(audio_rows),),2),torch.zeros(len(video_rows),dtype=torch.long)]).to(device)
    row_t=torch.full((len(position_ids),),float(t_v),device=device)
    row_t[state_indices]=1.;row_t[audio_indices]=t_a
    timestep,timestep_indices=torch.unique(row_t,sorted=True,return_inverse=True)
    audio_mask=audio_valid_mask[audio_cut:target_audio_end].to(device).repeat(2)
    if audio_mask.shape!=(len(audio_rows),):raise ValueError("Audio validity and row packing disagree")
    if len(token_tags) != len(position_ids):
        raise ValueError(
            f"Variable packed rows mismatch: text={text_len} history={history_count} "
            f"audio_rows={len(audio_rows)} audio_pos={len(current_audio_pos)} "
            f"video_rows={len(video_rows)} video_pos={len(current_video_pos)} "
            f"tags={len(token_tags)} positions={len(position_ids)}"
        )
    return dict(inputs=dict(hidden_states=video_rows[None],audio_hidden_states=audio_rows[None],
        encoder_hidden_states=prompt.to(device,torch.bfloat16)[None],timestep=timestep,timestep_indices=timestep_indices,
        token_tags=token_tags,position_ids=position_ids,video_indices=video_indices,audio_indices=audio_indices,
        text_indices=text_indices,state_indices=state_indices,return_dict=False),history_rows=clean_history_rows[None],
        prefix_rows=prefix,target_video=target_v,target_audio=target_a,audio_loss_mask=audio_mask,
        contract={"history_source":"same_episode_ground_truth_past_only","training_kind":"teacher_forced_conditional_velocity",
          "history_start_latent":history_start_latents,"history_end_latent":history_end_latents,"target_start_latent":target_start_latents,"target_end_latent":target_end,
          "past_pixel_frames":cut_pixel_frame,"audio_cut_latent":audio_cut,
          "target_video_latents":target_end-target_start_latents,"target_audio_latents":target_audio_end-audio_cut,
          "history_window_policy":"contiguous_recent_window" if history_frames<=0 else "sampled_history_candidate",
          "history_frame_ids":history_ids.tolist(),"history_tokens":history_count,
          "history_last_source_pixel_frame":17*((history_end_latents-1)//5)+4*((history_end_latents-1)%5) if history_end_latents else None,
          "current_start_pixel_frame":cut_pixel_frame,"sigma_video":float(sigma_v),"sigma_audio":float(sigma_a),
          "history_audio_used":False,"future_clean_target_used_as_input":False,"generated_history_training":False})


def velocity_loss(pred_video,pred_audio,batch):
    lv=F.mse_loss(pred_video[0].float(),batch["target_video"])
    mask=batch["audio_loss_mask"]
    if not mask.any():raise ValueError("No valid audio target rows")
    la=F.mse_loss(pred_audio[0,mask].float(),batch["target_audio"][mask])
    return lv+la,lv,la
