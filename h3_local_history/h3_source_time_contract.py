"""Generated-history packing on annotation windows and measured source PTS.

This is a source-clock execution interface, not formal encoder/event admission.
No source pixels, source audio or reference latents are read by this module.
"""
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_case(path):
    case = json.loads(Path(path).read_text())
    if case.get('schema') != 'h3_source_time_case_v1' or case.get('formal_source_time_admitted') is not False:
        raise ValueError('Use an explicitly unadmitted source-clock execution case')
    program = case['program']
    if case.get('seed')!=program['seed']:
        raise ValueError('Case seed and frozen source program differ')
    if (program.get('schema') != 'h3_source_time_program_v1' or
            program['program_sha256'] != digest({k:v for k,v in program.items() if k != 'program_sha256'})):
        raise ValueError('Frozen source program hash changed')
    if (program['sampling_source_audio_condition'] or program['audio_history_used'] or
            program['sampling_audio_future_context_allowed'] or program['sampling_audio_encoder_lookahead_sec'] != 0.):
        raise ValueError('This entry has no source audio encoder or audio-history input')
    validate_program(program)
    for clip in program['segments']:
        prompt = Path(clip['prompt_path'])
        if not prompt.is_absolute() or hashlib.sha256(prompt.read_bytes()).hexdigest() != clip['prompt_sha256']:
            raise ValueError('Frozen source-window plaintext embedding changed')
    return case, program


def validate_program(program):
    """Validate timeline and history contents, not just a caller-supplied hash."""
    def boundary(i):
        return 0 if i==0 else 5+17*((i-2)//5)
    segments=program['segments'];rows=program['generation_chunks'];expected=[];offset=0
    budget=program['history_h_max_sec']
    if not math.isfinite(budget) or budget<=0 or not segments or type(program.get('seed')) is not int or program['seed']<0:
        raise ValueError('Source history budget and windows required')
    delivered=[]
    for ci,clip in enumerate(segments):
        pts=clip['source_pts'];model=clip['model_pts'];frames=clip['native_output_frames'];latents=clip['video_latent_count']
        count=clip['valid_output_frames'];start,end=clip['source_start_sec'],clip['source_end_sec']
        if (count!=len(pts) or count<1 or frames<count or frames<22 or (frames-5)%17 or
            latents!=2+5*((frames-5)//17) or clip['padding_tail_frames']!=frames-count or
            len(model)!=frames+1 or model[:count]!=pts or
            clip['model_valid_mask']!=[i<count for i in range(len(model))] or
            any(not math.isfinite(t) for t in model) or any(a>=b for a,b in zip(model,model[1:])) or
            not start<=pts[0]<=pts[-1]<end or model[-1]<end or
            (ci and abs(start-segments[ci-1]['source_end_sec'])>1e-7)):
            raise ValueError('Source PTS, semantic windows or model padding changed')
        delivered.extend(pts)
        for a in [0]+list(range(7,latents,5)):
            b=7 if a==0 else a+5
            expected.append((ci,a,b,offset+a,offset+b))
        offset+=latents
    if (delivered!=program['actual_source_pts'] or program['actual_output_frames']!=len(delivered) or
        len(rows)!=len(expected) or program['source_time_window']!=[segments[0]['source_start_sec'],segments[-1]['source_end_sec']]):
        raise ValueError('Incomplete source presentation or generation calls')
    prior=[]
    for number,(row,indices) in enumerate(zip(rows,expected)):
        ci,a,b,ga,gb=indices;clip=segments[ci];start=clip['source_start_sec'];model=clip['model_pts']
        now=start if a==0 else model[boundary(a)];model_end=model[boundary(b)]
        support=max(17*(i//5)+4*(i%5) for i in range(a,b));eligible=support<clip['valid_output_frames']
        chosen=[p for p in prior if p['history_write_eligible'] and p['source_support_end_sec']<=now+1e-9
                and p['source_target_start_sec']>=now-budget-1e-9]
        parts=[dict(clip_index=p['clip_index'],local_start=p['clip_video_latent_start'],local_end=p['clip_video_latent_end']) for p in chosen]
        ids=[i for p in chosen for i in range(p['video_latent_start'],p['video_latent_end'])]
        actual=(row['clip_index'],row['clip_video_latent_start'],row['clip_video_latent_end'],row['video_latent_start'],row['video_latent_end'])
        if (actual!=indices or row['chunk']!=number or row['source_target_start_sec']!=now or
            row['model_target_end_sec']!=model_end or row['history_write_eligible'] is not eligible or
            row['source_support_end_sec']!=(model[support] if eligible else None) or row['history_parts']!=parts or
            row['history_frame_ids']!=ids or row['audio_latent_start']!=round((now-start)*40) or
            row['audio_latent_end']!=round((model_end-start)*40) or row['audio_latent_end']<=row['audio_latent_start'] or
            row['audio_latent_end']>clip['audio_latent_count']):
            raise ValueError('Future/padded history, changed coordinates or audio boundaries')
        prior.append(row)


def key(row):
    return row['clip_index'], row['clip_video_latent_start'], row['clip_video_latent_end']


class SourceHistory:
    def __init__(self, program):
        self.rows = program['generation_chunks']
        self.parts = {}
        self.next_chunk = 0

    def select(self, row):
        if self.next_chunk >= len(self.rows) or row != self.rows[self.next_chunk]:
            raise ValueError('Only the next declared source-time chunk may read history')
        wanted = {(p['clip_index'], p['local_start'], p['local_end']) for p in row['history_parts']}
        if wanted != set(self.parts):
            raise ValueError('Missing, padded, future or evicted source-time history')
        return self.parts

    def commit(self, row, value):
        import torch
        self.select(row)
        count = row['clip_video_latent_end'] - row['clip_video_latent_start']
        if value.ndim != 4 or value.shape[:2] != (24, count) or not torch.isfinite(value).all():
            raise ValueError('Invalid generated source-time chunk')
        if row['history_write_eligible']:
            self.parts[key(row)] = value.detach().cpu().clone()
        self.next_chunk += 1
        wanted = set()
        if self.next_chunk < len(self.rows):
            wanted = {(p['clip_index'], p['local_start'], p['local_end'])
                      for p in self.rows[self.next_chunk]['history_parts']}
        if wanted-set(self.parts):
            raise ValueError('Next chunk would read an uncommitted or padding-supported group')
        evicted = [k for k in self.parts if k not in wanted]
        for k in evicted:
            del self.parts[k]
        return evicted

    @property
    def bytes(self):
        return sum(v.numel()*v.element_size() for v in self.parts.values())


def source_batch(row, program, prompt, tags, generator, device, state, h=48, w=84, zero_history=False):
    import torch
    from h3_spatial_history_flow import matched_spatial_batch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from h3_native_shot_contract import nominal_frame
    if tags.shape != (prompt.shape[0],) or not bool((tags == 1).all()) or not torch.isfinite(prompt).all():
        raise ValueError('Finite plaintext conditions required')
    past = state.select(row)
    ci = row['clip_index']
    clip = program['segments'][ci]
    video = torch.zeros(24, clip['video_latent_count'], h, w, dtype=torch.bfloat16)
    audio = torch.zeros(2, 32, clip['audio_latent_count'], dtype=torch.bfloat16)
    start, end = row['clip_video_latent_start'], row['clip_video_latent_end']
    batch = matched_spatial_batch(video, audio, prompt, tags, torch.ones(audio.shape[-1], dtype=torch.bool),
        generator, device, history_frames=0, history_start_latents=0, history_end_latents=0,
        target_start_latents=start, target_end_latents=end,
        target_audio_start_latents=row['audio_latent_start'], target_audio_end_latents=row['audio_latent_end'])
    inputs = batch['inputs']
    old = inputs['position_ids']
    text_count, audio_count = inputs['text_indices'].numel(), inputs['audio_indices'].numel()
    per_frame = h*w//4
    grid = old[inputs['video_indices'][:per_frame], 1:]
    origin = program['source_time_window'][0]

    def positions(index, ids):
        c = program['segments'][index]
        times = [c['model_pts'][nominal_frame(i)]-origin for i in ids]
        clock = text_count+40*torch.tensor(times, device=device, dtype=old.dtype)
        result = old.new_empty((len(ids), per_frame, 3))
        result[:, :, 0] = clock[:, None]
        result[:, :, 1:] = grid[None]
        return result.reshape(-1, 3)

    values, pos = [], []
    for part in row['history_parts']:
        index, a, b = part['clip_index'], part['local_start'], part['local_end']
        v = past[index, a, b]
        if v.shape != (24, b-a, h, w):
            raise ValueError('Source history spatial shape differs')
        values.append(patchify_video_latents(v[None].to(device, torch.float32), (1,2,2)))
        pos.append(positions(index, range(a,b)))
    history = torch.cat(values)[None] if values else torch.empty((1,0,96), device=device, dtype=torch.float32)
    history_pos = torch.cat(pos) if pos else old.new_empty((0,3))
    audio_pos = old[inputs['audio_indices']].clone()
    clock = (text_count+40*(clip['generation_audio_anchor_sec']-origin) +
             torch.arange(row['audio_latent_start'], row['audio_latent_end'], device=device, dtype=old.dtype))
    audio_pos[:,0] = clock.repeat(2)
    prefix = text_count+history.shape[1]
    inputs['position_ids'] = torch.cat((old[inputs['text_indices']], history_pos, audio_pos, positions(ci,range(start,end))))
    inputs['state_indices'] = torch.arange(text_count, prefix, device=device)
    inputs['audio_indices'] = torch.arange(prefix, prefix+audio_count, device=device)
    inputs['video_indices'] = torch.arange(prefix+audio_count, len(inputs['position_ids']), device=device)
    inputs['token_tags'] = torch.cat((tags.to(device), torch.zeros(history.shape[1], device=device, dtype=torch.long),
        torch.full((audio_count,),2,device=device,dtype=torch.long), torch.zeros((end-start)*per_frame,device=device,dtype=torch.long)))
    tv, ta = 1-batch['contract']['sigma_video'], 1-batch['contract']['sigma_audio']
    times = torch.full((len(inputs['position_ids']),), tv, device=device)
    times[inputs['state_indices']] = 1.
    times[inputs['audio_indices']] = ta
    inputs['timestep'], inputs['timestep_indices'] = torch.unique(times, sorted=True, return_inverse=True)
    batch.update(history_rows=history, prefix_rows=prefix, sampling=True, zero_history_state=zero_history)
    del batch['target_video'], batch['target_audio']
    batch['contract'].update(history_source='own_committed_padding_free_source_time_groups',
        history_frame_ids=row['history_frame_ids'], clean_source_video_or_audio_read=False,
        absolute_av_clock='measured_source_video_pts_and_explicit_40Hz_audio_minus_source_origin',
        source_reference_encoder_mask_validated=False, formal_source_time_admitted=False)
    return batch
