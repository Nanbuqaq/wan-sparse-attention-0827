"""Native H3 geometry for benchmark scripts, independent of UI row limits."""
import copy
import hashlib
import json
from pathlib import Path


def program(segments, fps=24):
    if not segments or fps != 24:
        raise ValueError('This native H3 benchmark contract uses 24fps')
    plan, unique, end = [], [], 0
    for i, row in enumerate(segments):
        text, units = row['prompt'], row['units']
        if not isinstance(text,str) or not text.strip() or not isinstance(units,int) or not 1<=units<=8:
            raise ValueError('Each prompt span requires 1..8 native duration units')
        if text not in unique:unique.append(text)
        start=end;end+=5*units+(2 if i==0 else 0)
        a=0 if i==0 else 5+17*((start-2)//5);b=5+17*((end-2)//5)
        plan.append(dict(prompt=text,shot_id=str(row['shot_id']),chunks=units,shot=i,prompt_index=unique.index(text),
                         video_latent_start=start,video_latent_end=end,start_frame=a,end_frame=b,
                         start_sec=a/fps,end_sec=b/fps,audio_latent_start=round(a*40/fps),audio_latent_end=round(b*40/fps)))
    canonical=dict(schema='h3_benchmark_program_v1',prompts=unique,plan=plan,frames=plan[-1]['end_frame'],
                   fps=fps,duration_sec=plan[-1]['end_sec'],clock='explicit_generated_clock_not_source_PTS',
                   unit_frames=17,initial_extra_frames=5,segments=segments)
    canonical['program_sha256']=hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return canonical


def materialize(value, prompt_paths, out, seed, chunk_policy='variable'):
    if chunk_policy not in ('variable','fixed5'):raise ValueError(chunk_policy)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    count=value['plan'][-1]['video_latent_end'];frames=value['frames'];sid='generated_'+value['program_sha256'][:12]
    source=dict(time_zero_sec=0.,audio_origin_sec=0.,source_frame_interval_end_sec=value['duration_sec'],
                video_pts=[i/24 for i in range(frames+8)],video_latent_count=count,audio_latent_count=round(frames*40/24),
                kept_video_latent_intervals=[[0,7]]+[[i,i+5] for i in range(7,count,5)],
                native_output_frames=frames,valid_output_frames=frames,padding_output_frames=0,source_pts=None,
                output_timestamps=dict(fps=24,kind='generated_clock'),timeline_kind='synthetic_generation_clock_not_source_media',data_version=None)
    chunks=[]
    for row in value['plan']:
        p=Path(prompt_paths[row['prompt']])
        if not p.is_file():raise FileNotFoundError(p)
        starts=[row['video_latent_start']]
        if chunk_policy=='fixed5':
            first=7 if starts[0]==0 else starts[0]+5
            starts+=list(range(first,row['video_latent_end'],5))
        for number,a in enumerate(starts):
            b=starts[number+1] if number+1<len(starts) else row['video_latent_end']
            af=0 if a==0 else 5+17*((a-2)//5);bf=5+17*((b-2)//5)
            chunks.append(dict(window_id=f"span{row['shot']}_part{number}",parent_window_id='shot_'+row['shot_id'],
                               chunk_kind='fixed5_startup7' if chunk_policy=='fixed5' else 'variable_prompt_span',prompt_path=str(p),
                               video_latent_start=a,video_latent_end=b,audio_latent_start=round(af*40/24),audio_latent_end=round(bf*40/24),
                               source_target_start_sec=af/24,source_target_end_sec=bf/24,
                               demo_units=(b-a-2)//5 if a==0 else (b-a)//5,program_prompt_index=row['prompt_index']))
    case=dict(case_id=sid,source_id=sid,seed=seed,nominal_output_fps=24,chunks=chunks,program_sha256=value['program_sha256'],
              generation_demo_only=True,chunk_policy=chunk_policy)
    for name,data in [('GENERATION_CASES.json',dict(cases=[case])),('SOURCE_CONTRACTS.json',{sid:source}),('PROGRAM.json',value)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return case,source
