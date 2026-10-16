"""Prompt/count program and a 24 fps *generated* clock; no source-data claim."""
import hashlib
import json
from pathlib import Path

FPS = 24
MAX_UNITS = 40


def normalize(payload):
    rows = payload.get('shots') if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows or len(rows) > 20:
        raise ValueError('请输入 1–20 行镜头。')
    merged = []
    total = 0
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) == 2:
            prompt, count = row
        elif isinstance(row, dict):
            prompt, count = row.get('prompt'), row.get('chunks')
        else:
            raise ValueError('每行需要 prompt 和 chunks。')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
            raise ValueError('提示词不能为空，且不能超过 12000 字符。')
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_UNITS:
            raise ValueError('段数必须是 1–40 的整数。')
        prompt = prompt.strip()
        total += count
        if total > MAX_UNITS:
            raise ValueError('当前 demo 一次最多 40 段。')
        if merged and merged[-1]['prompt'] == prompt:
            merged[-1]['chunks'] += count
        else:
            merged.append(dict(prompt=prompt, chunks=count))
    unique = list(dict.fromkeys(s['prompt'] for s in merged))
    video_end = 0
    plan = []
    for i, row in enumerate(merged):
        vstart = video_end
        video_end += 5 * row['chunks'] + (2 if i == 0 else 0)
        start_frame = 0 if i == 0 else 5 + 17 * ((vstart - 2) // 5)
        end_frame = 5 + 17 * ((video_end - 2) // 5)
        plan.append(dict(**row, shot=i, prompt_index=unique.index(row['prompt']),
            video_latent_start=vstart, video_latent_end=video_end,
            start_frame=start_frame, end_frame=end_frame,
            start_sec=start_frame/FPS, end_sec=end_frame/FPS,
            audio_latent_start=round(start_frame*40/FPS), audio_latent_end=round(end_frame*40/FPS)))
    canonical = dict(schema='h3_prompt_count_demo_v1', shots=merged, fps=FPS,
        unit_frames=17, initial_extra_frames=5, frames=plan[-1]['end_frame'],
        duration_sec=plan[-1]['end_sec'], total_units=total, plan=plan,
        prompts=unique, clock='synthetic_generation_24fps_not_dataset_source_PTS',
        generation='one variable span per merged prompt; not a fixed-chunk trained model')
    canonical['program_sha256'] = hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return canonical


def materialize(program, prompts_dir, out, seed):
    """CPU only. Prompt embeddings have already been encoded on an InferHub worker."""
    out = Path(out);out.mkdir(parents=True,exist_ok=True)
    count = program['plan'][-1]['video_latent_end']
    end_sec = program['duration_sec'];frames = program['frames']
    sid = 'generated_' + program['program_sha256'][:12]
    # Extra nominal timestamps support the codec's final partial group; they are not
    # observations and are explicitly synthetic generation coordinates.
    pts = [i/FPS for i in range(frames+8)]
    intervals = [[0,7]] + [[i,i+5] for i in range(7,count,5)]
    source = dict(time_zero_sec=0.,audio_origin_sec=0.,source_frame_interval_end_sec=end_sec,
        video_pts=pts,video_latent_count=count,audio_latent_count=round(frames*40/FPS),
        kept_video_latent_intervals=intervals,native_output_frames=frames,valid_output_frames=frames,
        padding_output_frames=0,source_pts=None,output_timestamps=dict(fps=FPS,kind='generated_clock'),
        timeline_kind='synthetic_generation_clock_not_source_media',data_version=None)
    chunks = []
    for s in program['plan']:
        prompt_path = Path(prompts_dir)/f"prompt_{s['prompt_index']}.pt"
        if not prompt_path.is_file():raise FileNotFoundError(prompt_path)
        chunks.append(dict(window_id=f"shot{s['shot']}",parent_window_id=f"shot{s['shot']}",
            chunk_kind='user_prompt_variable_span',prompt_path=str(prompt_path),
            video_latent_start=s['video_latent_start'],video_latent_end=s['video_latent_end'],
            audio_latent_start=s['audio_latent_start'],audio_latent_end=s['audio_latent_end'],
            source_target_start_sec=s['start_sec'],source_target_end_sec=s['end_sec'],
            demo_units=s['chunks'],program_prompt_index=s['prompt_index']))
    case=dict(case_id=sid,source_id=sid,seed=seed,nominal_output_fps=FPS,chunks=chunks,
        program_sha256=program['program_sha256'],generation_demo_only=True)
    for name,value in [('SOURCE_CONTRACTS.json',{sid:source}),('GENERATION_CASES.json',{'cases':[case]}),('PROGRAM_RESOLVED.json',program)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return case,source
