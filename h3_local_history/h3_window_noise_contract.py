"""Coordinate-preserving noise mapping for sequential and joint suffix replay.

No model is constructed here. Video tokens are frame-major; audio tokens are
channel-major, so concatenating the flat audio sequences would change the noise.
"""
import copy
import math

import torch


def joint_row(rows):
    if len(rows) != 2:
        raise ValueError('The diagnostic requires exactly two suffix chunks')
    first, second = rows
    for key in ('parent_window_id', 'program_prompt_index'):
        if first[key] != second[key]:
            raise ValueError('Joint window would cross a semantic prompt boundary')
    for key in ('video', 'audio'):
        if first[f'{key}_latent_end'] != second[f'{key}_latent_start']:
            raise ValueError('Suffix has a noncontiguous modality interval')
    if any(r['video_latent_end'] - r['video_latent_start'] != 5 for r in rows):
        raise ValueError('This experiment compares 5+5 against 10 video latents')
    if not math.isclose(first['source_target_end_sec'], second['source_target_start_sec'], abs_tol=1e-8):
        raise ValueError('Suffix time windows are not contiguous')
    result = copy.deepcopy(first)
    for key in ('video_latent_end', 'audio_latent_end', 'source_target_end_sec'):
        result[key] = second[key]
    result.update(window_id=first['window_id'] + '_joint10',
                  chunk_kind='joint10_suffix_diagnostic', demo_units=2)
    return result


def combine_noise(references, rows):
    joint_row(rows)
    if len(references) != len(rows):
        raise ValueError('Noise references do not cover the suffix')
    video, audio, audio_lengths = [], [], []
    for reference, row in zip(references, rows):
        if reference['current_chunk'] != row:
            raise ValueError('Saved noise geometry differs from the requested chunk')
        v, a = reference['initial_video_noise'], reference['initial_audio_noise']
        count = row['audio_latent_end'] - row['audio_latent_start']
        if (v.ndim != 3 or v.shape[0] != 1 or v.shape[2] != 96 or
                v.shape[1] % 5 or a.shape != (1, 2 * count, 32) or
                v.dtype != torch.float32 or a.dtype != torch.float32):
            raise ValueError('Unexpected saved noise layout or precision')
        if not torch.isfinite(v).all() or not torch.isfinite(a).all():
            raise ValueError('Nonfinite saved noise')
        video.append(v)
        audio.append(a.reshape(1, 2, count, 32))
        audio_lengths.append(count)
    if video[0].shape[1] != video[1].shape[1]:
        raise ValueError('Spatial token layout differs between suffix chunks')
    return torch.cat(video, dim=1), torch.cat(audio, dim=2).reshape(1, 2 * sum(audio_lengths), 32)


def split_noise(video, audio, rows):
    joint_row(rows)
    if video.ndim != 3 or video.shape[0] != 1 or video.shape[1] % 10 or video.shape[2] != 96:
        raise ValueError('Unexpected joint video noise')
    sizes = [r['audio_latent_end'] - r['audio_latent_start'] for r in rows]
    if audio.shape != (1, 2 * sum(sizes), 32):
        raise ValueError('Unexpected joint audio noise')
    videos = video.chunk(2, dim=1)
    audios = audio.reshape(1, 2, sum(sizes), 32).split(sizes, dim=2)
    return [(v.contiguous(), a.reshape(1, -1, 32).contiguous()) for v, a in zip(videos, audios)]
