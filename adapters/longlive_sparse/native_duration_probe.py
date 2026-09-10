"""Isolate elapsed generation/away duration without changing Q, K or archive count."""
import math
import torch


def duration_geometry(length,base_length):
    if base_length not in (64,128) or length<base_length or length%8:
        raise ValueError('duration probe requires a block8 multiple no shorter than its base')
    return dict(latent_frames=length,pixel_frames=4*length-3,fps=24,
        media_duration_s=(4*length-3)/24,chunk_frames=8,base_latent_frames=base_length,
        archived_scene_count_unchanged=True,actual_attention_K_unchanged_by_duration=True,
        scripted_generated_history_not_natural_access_trace=True)


def stretch_away_schedule(segments,base_prompts,length,base_length):
    duration_geometry(length,base_length)
    if len(base_prompts)!=1 or len(base_prompts[0])!=base_length//8:
        raise ValueError('base prompt/block geometry mismatch')
    if len(segments)!=4 or segments[2]['role']!='away' or segments[3]['role']!='return_without_restatement':
        raise ValueError('qualified source/away/return schedule required')
    if segments[3]['start_latent']%8 or segments[3]['start_latent']>=base_length:
        raise ValueError('invalid original return boundary')
    new_segments=[dict(s) for s in segments]
    new_segments[-1]['start_latent']+=length-base_length
    boundary=segments[-1]['start_latent']//8
    # Reuse the already-expanded ordinary away prompt; never add extra cuts.
    ordinary_away=base_prompts[0][boundary-1]
    if ordinary_away.startswith('The scene transitions. '):
        raise ValueError('away extension must not repeat a scene-cut prefix')
    prompts=base_prompts[0][:boundary]+[ordinary_away]*((length-base_length)//8)+base_prompts[0][boundary:]
    return new_segments,[prompts]


def duration_noise(shape,*,base_length,seed,device,dtype=torch.bfloat16,alignment='absolute',return_base=False):
    """Exact original base draw, then independent fixed-size tail draws.

    Longer full-tensor CUDA randn calls need not preserve shorter-tensor prefixes.
    Tail generation uses a separate generator and cannot advance the global RNG
    used by the native denoiser. No already-generated latent/KV is repeated.
    """
    if len(shape)!=5 or shape[0]!=1 or min(shape)<=0:
        raise ValueError('positive native batch1 video shape required')
    if alignment not in ('absolute','return_event'):raise ValueError('unknown duration noise alignment')
    duration_geometry(shape[1],base_length)
    prefix_shape=(shape[0],base_length,*shape[2:])
    prefix=torch.randn(prefix_shape,device=device,dtype=dtype)
    if shape[1]==base_length:return (prefix,prefix) if return_base else prefix
    generator=torch.Generator(device=device).manual_seed(int(seed)^0x5DEECE66D)
    chunks=[]
    for _ in range(math.ceil((shape[1]-base_length)/base_length)):
        chunks.append(torch.randn(prefix_shape,device=device,dtype=dtype,generator=generator))
    if alignment=='absolute':
        output=torch.cat([prefix,*chunks],dim=1)[:,:shape[1]].contiguous()
    else:
        return_frames=base_length//4
        extension=torch.cat(chunks,dim=1)[:,:shape[1]-base_length]
        output=torch.cat([prefix[:,:base_length-return_frames],extension,prefix[:,base_length-return_frames:]],dim=1)
    return (output,prefix) if return_base else output
