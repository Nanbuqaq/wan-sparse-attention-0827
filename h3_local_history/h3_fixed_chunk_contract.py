"""Fixed H3 chunks with explicit source-time gaps and generated-history parity."""
import torch
from h3_fsdp2_speed import SpeedTrainingModel


class FixedTrainingModel(SpeedTrainingModel):
    def forward(self,batch):
        if not batch.get('sampling',False):
            return super().forward(batch)
        inputs=dict(batch['inputs'])
        inputs['state_hidden_states']=self.projection(batch['history_rows'])
        if batch.get('zero_history_state',False):
            inputs['state_hidden_states']=torch.zeros_like(inputs['state_hidden_states'])
        return self.transformer(**inputs)


def pixel_boundary(latent):
    if latent == 0:
        return 0
    if latent < 2 or (latent - 2) % 5:
        raise ValueError(f'Illegal native boundary {latent}')
    return 5 + 17 * ((latent - 2) // 5)


def nominal_frame(latent):
    return 17 * (latent // 5) + (0, 1, 5, 9, 13)[latent % 5]


def support_last_frame(latent):
    return 17 * (latent // 5) + 4 * (latent % 5)


def history_ids(source, target_start, h_max_sec=8.):
    pts = source['video_pts']
    boundary=pixel_boundary(target_start)
    now = pts[boundary] if boundary<len(pts) else source['source_frame_interval_end_sec']
    chosen = []
    for start, end in source['kept_video_latent_intervals']:
        if end > target_start:
            continue
        # Keep whole native groups. The startup exception is also indivisible.
        if pts[pixel_boundary(start)] < now - h_max_sec - 1e-7:
            continue
        for i in range(start, end):
            if support_last_frame(i) >= len(pts) or pts[support_last_frame(i)] >= now:
                raise ValueError('History latent includes an unobserved source frame.')
            chosen.append(i)
    return sorted(set(chosen))


def fixed_matched_batch(row, source, video, audio, prompt, tags, mask, generator,
                        device, h_max_sec=8., retained_ids=None,retained_video=None):
    from h3_spatial_history_flow import matched_spatial_batch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    start, end = int(row['video_latent_start']), int(row['video_latent_end'])
    chosen = history_ids(source, start, h_max_sec) if retained_ids is None else list(retained_ids)
    if chosen != sorted(set(chosen)) or any(i < 0 or i >= start for i in chosen):
        raise ValueError('History must contain unique ordered past latent IDs.')
    allowed = set(history_ids(source, start, h_max_sec))
    if not set(chosen) <= allowed:
        raise ValueError('History reads discarded, future, or out-of-budget frames.')
    batch = matched_spatial_batch(video,audio,prompt,tags,mask,generator,device,
        history_frames=0,history_start_latents=0,history_end_latents=0,
        target_start_latents=start,target_end_latents=end,
        target_audio_start_latents=int(row['audio_latent_start']),
        target_audio_end_latents=int(row['audio_latent_end']))
    inputs=batch['inputs'];old_pos=inputs['position_ids']
    text_count=inputs['text_indices'].numel()
    audio_count=inputs['audio_indices'].numel()
    rows_per_frame=(video.shape[-2]//2)*(video.shape[-1]//2)
    frame_grid=old_pos[inputs['video_indices'][:rows_per_frame],1:]
    pts=source['video_pts'];zero=source['time_zero_sec']
    def positions(ids):
        clock=torch.tensor([text_count+40*(pts[nominal_frame(i)]-zero) for i in ids],
                           device=device,dtype=old_pos.dtype)
        result=torch.empty((len(ids),rows_per_frame,3),device=device,dtype=old_pos.dtype)
        result[:,:,0]=clock[:,None];result[:,:,1:]=frame_grid[None]
        return result.reshape(-1,3)
    current_video_pos=positions(range(start,end))
    current_audio_pos=old_pos[inputs['audio_indices']].clone()
    audio_clock=text_count+40*(source['audio_origin_sec']-zero)+torch.arange(
        int(row['audio_latent_start']),int(row['audio_latent_end']),device=device,dtype=old_pos.dtype)
    current_audio_pos[:,0]=audio_clock.repeat(2)
    if chosen:
        values=video.index_select(1,torch.tensor(chosen)) if retained_video is None else retained_video
        if values.shape!=(24,len(chosen),video.shape[-2],video.shape[-1]):
            raise ValueError('Explicit generated history does not match its frame IDs.')
        values=values.to(device,torch.float32)
        past=patchify_video_latents(values[None],(1,2,2))[None]
    else:
        past=torch.empty((1,0,96),device=device,dtype=torch.float32)
    prefix=text_count+past.shape[1]
    inputs['position_ids']=torch.cat((old_pos[inputs['text_indices']],positions(chosen),current_audio_pos,current_video_pos))
    inputs['state_indices']=torch.arange(text_count,prefix,device=device)
    inputs['audio_indices']=torch.arange(prefix,prefix+audio_count,device=device)
    inputs['video_indices']=torch.arange(prefix+audio_count,len(inputs['position_ids']),device=device)
    inputs['token_tags']=torch.cat((tags.to(device),torch.zeros(past.shape[1],device=device,dtype=torch.long),
        torch.full((audio_count,),2,device=device,dtype=torch.long),
        torch.zeros((end-start)*rows_per_frame,device=device,dtype=torch.long)))
    tv=1.-batch['contract']['sigma_video'];ta=1.-batch['contract']['sigma_audio']
    row_t=torch.full((len(inputs['position_ids']),),tv,device=device)
    row_t[inputs['state_indices']]=1.;row_t[inputs['audio_indices']]=ta
    inputs['timestep'],inputs['timestep_indices']=torch.unique(row_t,sorted=True,return_inverse=True)
    batch['history_rows']=past;batch['prefix_rows']=prefix
    batch['contract'].update(history_frame_ids=chosen,history_tokens=past.shape[1],
        history_start_latent=min(chosen) if chosen else start,history_end_latent=start,
        history_window_policy='whole_retained_chunks_in_source_time_budget',
        discarded_intervals_excluded_from_history=True,absolute_av_clock='actual_source_pts_40hz',
        parent_window_id=row['parent_window_id'],chunk_kind=row['chunk_kind'],
        source_target_start_sec=row['source_target_start_sec'],source_target_end_sec=row['source_target_end_sec'])
    return batch
