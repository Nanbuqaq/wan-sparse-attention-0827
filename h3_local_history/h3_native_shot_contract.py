"""Generated-only native-shot clock/state packing, derived from the teacher entry.

Each shot has its own 7-latent startup. Global video IDs track lineage, while
positions use each shot's local native clock plus its episode time offset.
Only explicitly committed generated chunks can enter history.
"""
import hashlib
import json
from pathlib import Path

import torch


def pixel_boundary(latent):
    if latent == 0:
        return 0
    if latent < 2 or (latent-2) % 5:
        raise ValueError('Illegal native boundary')
    return 5+17*((latent-2)//5)


def nominal_frame(latent):
    return 17*(latent//5)+(0, 1, 5, 9, 13)[latent % 5]


def support_last_frame(latent):
    return 17*(latent//5)+4*(latent % 5)


def load_case(path):
    raw=Path(path).read_bytes();case=json.loads(raw)
    if case.get('schema')!='h3_native_shot_case_v1' or case.get('audio_history_used',False):
        raise ValueError('Use a native-shot plaintext case without audio history')
    if case.get('history_h_max_sec')!=8. or not isinstance(case.get('seed'),int):
        raise ValueError('Declare the eight-second history budget and seed')
    clips=[];offset=0.
    for item in case['clips']:
        frames=item['num_frames']
        if not isinstance(frames,int) or isinstance(frames,bool) or frames<22 or (frames-5)%17:
            raise ValueError('Each shot requires 5+17*n native frames')
        prompt=Path(item['prompt_path'])
        if not prompt.is_absolute() or hashlib.sha256(prompt.read_bytes()).hexdigest()!=item['prompt_sha256']:
            raise ValueError('Frozen plaintext prompt changed')
        denied='generation-only://'+case['case_id']+'/'+item['clip_id']+'/'
        clips.append(dict(clip_id=item['clip_id'],num_frames=frames,fps=24.,offset_sec=offset,
                          video_latent_count=2+5*((frames-5)//17),student_prompt_path=str(prompt),
                          student_prompt_sha256=item['prompt_sha256'],video_latents=denied+'video',
                          audio_latents=denied+'audio',audio_valid_mask=denied+'mask'))
        offset+=frames/24.
    episode=dict(schema='h3_native_shot_generation_case_v1',episode_id=case['case_id'],
                 source_group='generation_only:'+case['case_id'],split='generation_only',training_ready=False,
                 event_eval_ready=False,generation_only=True,clips=clips)
    native_rows(episode)
    return case,episode


def native_rows(episode,h_max_sec=8.):
    if h_max_sec<=0:
        raise ValueError('History budget must be positive')
    rows=[];video_offset=0;expected_offset=0.;seen=set()
    for ci,clip in enumerate(episode['clips']):
        count,fps=int(clip['num_frames']),float(clip['fps'])
        if fps!=24. or count<22 or (count-5)%17:
            raise ValueError('Require native 24FPS shot geometry')
        if abs(clip['offset_sec']-expected_offset)>1e-7 or clip['clip_id'] in seen:
            raise ValueError('Shot offsets must be contiguous and IDs unique')
        seen.add(clip['clip_id']);latent_count=2+5*((count-5)//17)
        if clip['video_latent_count']!=latent_count:
            raise ValueError('Native latent count differs')
        for start in [0]+list(range(7,latent_count,5)):
            end=7 if start==0 else start+5
            now=expected_offset+pixel_boundary(start)/fps
            end_sec=expected_offset+pixel_boundary(end)/fps
            chosen=[];parts=[]
            for prior in rows:
                if prior['source_target_end_sec']>now+1e-7 or prior['source_target_start_sec']<now-h_max_sec-1e-7:
                    continue
                previous=episode['clips'][prior['clip_index']]
                for li in range(prior['clip_video_latent_start'],prior['clip_video_latent_end']):
                    if previous['offset_sec']+support_last_frame(li)/fps>=now-1e-7:
                        raise ValueError('History includes an unobserved video frame')
                chosen.extend(range(prior['video_latent_start'],prior['video_latent_end']))
                parts.append(dict(clip_index=prior['clip_index'],local_start=prior['clip_video_latent_start'],local_end=prior['clip_video_latent_end']))
            rows.append(dict(window_id=f'{episode["episode_id"]}:{clip["clip_id"]}:{start:04d}',
                source_id=episode['episode_id'],sample_id=episode['episode_id'],source_group=episode['source_group'],split=episode['split'],
                clip_index=ci,clip_video_latent_start=start,clip_video_latent_end=end,
                video_latent_start=video_offset+start,video_latent_end=video_offset+end,
                audio_latent_start=round(pixel_boundary(start)*40/fps),audio_latent_end=round(pixel_boundary(end)*40/fps),
                source_target_start_sec=now,source_target_end_sec=end_sec,parent_window_id=f'{episode["episode_id"]}:{clip["clip_id"]}',
                video_path=clip['video_latents'],audio_path=clip['audio_latents'],audio_valid_mask_path=clip['audio_valid_mask'],
                prompt_path=clip['student_prompt_path'],history_frame_ids=chosen,history_parts=parts,
                chunk_kind='native_initial_7' if start==0 else 'regular_5',
                transition_kind=('initial' if not rows else 'cross_shot_first') if start==0 else 'same_shot_continuation',
                data_kind='teacher_native_clip_episode',data_version='h3_dense_teacher_native_clip_episode_v1',
                global_ids_are_lineage_only=True,native_boundary_policy='startup7_each_shot_then_regular5',
                audio_history_used=False,history_h_max_sec=h_max_sec,training_ready=episode.get('training_ready',False),event_eval_ready=False))
        video_offset+=latent_count;expected_offset+=count/fps
    return rows


class NativeHistory:
    def __init__(self,episode):
        self.rows=native_rows(episode);self.parts={};self.next_chunk=0;self.h_max_sec=8.

    def select(self,row):
        if self.next_chunk>=len(self.rows) or row!=self.rows[self.next_chunk]:
            raise ValueError('Read only the next declared generation chunk')
        wanted={(p['clip_index'],p['local_start'],p['local_end']) for p in row['history_parts']}
        if wanted!=set(self.parts):
            raise ValueError('History contains missing, future or evicted chunks')
        return {key:value['value'] for key,value in self.parts.items()}

    def commit(self,row,value):
        self.select(row)
        count=row['clip_video_latent_end']-row['clip_video_latent_start']
        if value.ndim!=4 or value.shape[:2]!=(24,count) or not torch.isfinite(value).all():
            raise ValueError('Invalid generated native chunk')
        key=(row['clip_index'],row['clip_video_latent_start'],row['clip_video_latent_end'])
        self.parts[key]=dict(value=value.detach().cpu().clone(),source_start_sec=row['source_target_start_sec'],
                            global_ids=list(range(row['video_latent_start'],row['video_latent_end'])))
        evicted=[]
        for key,part in list(self.parts.items()):
            if part['source_start_sec']<row['source_target_end_sec']-8.-1e-7:
                evicted.extend(part['global_ids']);del self.parts[key]
        self.next_chunk+=1
        return evicted

    @property
    def bytes(self):
        return sum(p['value'].numel()*p['value'].element_size() for p in self.parts.values())


def native_batch(row,episode,prompt,tags,generator,device,state,h=48,w=84,zero_history=False):
    from h3_spatial_history_flow import matched_spatial_batch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    if tags.shape!=(prompt.shape[0],) or not bool((tags==1).all()) or not torch.isfinite(prompt).all():
        raise ValueError('Only finite plaintext conditioning is accepted')
    past=state.select(row);ci=row['clip_index'];clip=episode['clips'][ci]
    video=torch.zeros(24,clip['video_latent_count'],h,w,dtype=torch.bfloat16)
    audio=torch.zeros(2,32,round(clip['num_frames']/24.*40),dtype=torch.bfloat16)
    start,end=row['clip_video_latent_start'],row['clip_video_latent_end']
    batch=matched_spatial_batch(video,audio,prompt,tags,torch.ones(audio.shape[-1],dtype=torch.bool),generator,device,
        history_frames=0,history_start_latents=0,history_end_latents=0,target_start_latents=start,target_end_latents=end,
        target_audio_start_latents=row['audio_latent_start'],target_audio_end_latents=row['audio_latent_end'])
    inputs=batch['inputs'];old=inputs['position_ids'];text_count=inputs['text_indices'].numel();audio_count=inputs['audio_indices'].numel()
    per_frame=h*w//4;grid=old[inputs['video_indices'][:per_frame],1:]
    def positions(index,ids):
        c=episode['clips'][index]
        clock=torch.tensor([text_count+40*(c['offset_sec']+nominal_frame(i)/c['fps']) for i in ids],device=device,dtype=old.dtype)
        out=old.new_empty((len(ids),per_frame,3));out[:,:,0]=clock[:,None];out[:,:,1:]=grid[None]
        return out.reshape(-1,3)
    values=[];pos=[]
    for part in row['history_parts']:
        index,a,b=part['clip_index'],part['local_start'],part['local_end'];v=past[(index,a,b)]
        if v.shape!=(24,b-a,h,w):raise ValueError('Generated history spatial shape differs')
        values.append(patchify_video_latents(v[None].to(device,torch.float32),(1,2,2)))
        pos.append(positions(index,range(a,b)))
    history=torch.cat(values)[None] if values else torch.empty((1,0,96),device=device,dtype=torch.float32)
    history_pos=torch.cat(pos) if pos else old.new_empty((0,3))
    audio_pos=old[inputs['audio_indices']].clone()
    clock=text_count+40*clip['offset_sec']+torch.arange(row['audio_latent_start'],row['audio_latent_end'],device=device,dtype=old.dtype)
    audio_pos[:,0]=clock.repeat(2);prefix=text_count+history.shape[1]
    inputs['position_ids']=torch.cat((old[inputs['text_indices']],history_pos,audio_pos,positions(ci,range(start,end))))
    inputs['state_indices']=torch.arange(text_count,prefix,device=device)
    inputs['audio_indices']=torch.arange(prefix,prefix+audio_count,device=device)
    inputs['video_indices']=torch.arange(prefix+audio_count,len(inputs['position_ids']),device=device)
    inputs['token_tags']=torch.cat((tags.to(device),torch.zeros(history.shape[1],device=device,dtype=torch.long),
        torch.full((audio_count,),2,device=device,dtype=torch.long),torch.zeros((end-start)*per_frame,device=device,dtype=torch.long)))
    tv,ta=1-batch['contract']['sigma_video'],1-batch['contract']['sigma_audio']
    times=torch.full((len(inputs['position_ids']),),tv,device=device);times[inputs['state_indices']]=1.;times[inputs['audio_indices']]=ta
    inputs['timestep'],inputs['timestep_indices']=torch.unique(times,sorted=True,return_inverse=True)
    batch.update(history_rows=history,prefix_rows=prefix,sampling=True,zero_history_state=zero_history)
    del batch['target_video'],batch['target_audio']
    batch['contract'].update(history_source='committed_generated_native_shot_chunks_only',clean_teacher_video_or_audio_read=False,
                             history_frame_ids=row['history_frame_ids'],absolute_av_clock='explicit_episode_offset_plus_native_clip_40hz',
                             native_boundary_policy='startup7_each_shot_then_regular5')
    return batch
