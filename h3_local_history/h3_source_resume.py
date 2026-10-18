"""Source-clock committed-prefix identity, generated state and RNG restoration."""
import hashlib,os
from pathlib import Path
import torch
from h3_source_time_contract import SourceHistory,validate_program


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def save_progress(path,*,program,case_sha256,weights_spec_sha256,checkpoint_sha256,mode,
                  generated,records,generator,cuda_rng_state=None,execution_fingerprint=None):
    value=dict(schema='h3_source_progress_v2',program_sha256=program['program_sha256'],
        case_sha256=case_sha256,weights_spec_sha256=weights_spec_sha256,checkpoint_sha256=checkpoint_sha256,
        mode=mode,completed_chunks=len(records),generated_clips=generated,records=records,
        generator_state=generator.get_state().cpu(),cpu_rng_state=torch.get_rng_state(),
        cuda_rng_state=cuda_rng_state.cpu() if cuda_rng_state is not None else None,
        execution_fingerprint=execution_fingerprint,
        sampling_contract='source_time_native_padding_trim_v1_NFE50_CFG1',GPU_restore_equivalence_verified=False)
    path=Path(path);temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    torch.save(value,temporary);os.replace(temporary,path)


def restore(path,expected_sha256,*,program,case_sha256,weights_spec_sha256,checkpoint_sha256,
            mode,history,generator,set_cuda_rng_state=None,execution_fingerprint=None):
    validate_program(program)
    if history.next_chunk or history.parts:raise ValueError('Restore into an empty source history only')
    if file_sha(path)!=expected_sha256:raise ValueError('Source prefix archive checksum changed')
    value=torch.load(path,map_location='cpu',weights_only=True)
    expected=dict(schema='h3_source_progress_v2',program_sha256=program['program_sha256'],
        case_sha256=case_sha256,weights_spec_sha256=weights_spec_sha256,checkpoint_sha256=checkpoint_sha256,
        mode=mode,sampling_contract='source_time_native_padding_trim_v1_NFE50_CFG1',execution_fingerprint=execution_fingerprint)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Source prefix belongs to another input/model/policy')
    count=value['completed_chunks'];records=value['records'];generated=value['generated_clips'];rows=program['generation_chunks']
    if type(count) is not int or not 0<count<=len(rows) or len(records)!=count:raise ValueError('Invalid source committed prefix')
    rebuilt=SourceHistory(program);written={}
    for index,(row,record) in enumerate(zip(rows[:count],records)):
        if record.get('chunk')!=index or record['row']!=row or record.get('state_mode')!=mode or record['NFE']!=50 or record['CFG']!=1.:
            raise ValueError('Source prefix chunk records differ from requested policy')
        ci=row['clip_index'];clip=program['segments'][ci]
        if ci not in generated:raise ValueError('Missing source generated clip')
        video,audio=generated[ci]['video'],generated[ci]['audio']
        if (video.shape!=(24,clip['video_latent_count'],48,84) or audio.shape!=(2,32,clip['audio_latent_count']) or
            video.dtype!=torch.bfloat16 or audio.dtype!=torch.bfloat16):
            raise ValueError('Source prefix geometry or dtype changed')
        a,b=row['clip_video_latent_start'],row['clip_video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end']
        ends=written.setdefault(ci,[0,0])
        if [a,aa]!=ends:raise ValueError('Uncommitted gap in source prefix')
        if not torch.isfinite(video[:,a:b]).all() or not torch.isfinite(audio[:,:,aa:ae]).all():
            raise ValueError('Nonfinite committed source prefix')
        ends[:]=[b,ae];evicted=rebuilt.commit(row,video[:,a:b])
        if ([tuple(p) for p in evicted]!=[tuple(p) for p in record['evicted_parts']] or rebuilt.bytes!=record['history_state_bytes']):
            raise ValueError('Source history write/eviction differs from saved prefix')
    if set(generated)!=set(written):raise ValueError('Unobserved future clip in source prefix')
    for ci,(end,aend) in written.items():
        if torch.count_nonzero(generated[ci]['video'][:,end:]) or torch.count_nonzero(generated[ci]['audio'][:,:,aend:]):
            raise ValueError('Uncommitted future values in source prefix')
    # Validate RNG formats without mutating the caller before all data checks pass.
    try:
        candidate=torch.Generator(device=generator.device);candidate.set_state(value['generator_state'])
        cpu=torch.Generator(device='cpu');cpu.set_state(value['cpu_rng_state'])
    except (RuntimeError,TypeError) as error:raise ValueError('Source random-number state is invalid') from error
    cuda=value.get('cuda_rng_state')
    if set_cuda_rng_state is not None:
        if not isinstance(cuda,torch.Tensor) or cuda.dtype!=torch.uint8 or cuda.ndim!=1 or not cuda.numel():
            raise ValueError('Actual CUDA RNG state missing from source prefix')
        set_cuda_rng_state(cuda)
    elif cuda is not None:raise ValueError('Do not discard saved CUDA RNG state during restore')
    generator.set_state(value['generator_state']);torch.set_rng_state(value['cpu_rng_state'])
    history.parts=rebuilt.parts;history.next_chunk=rebuilt.next_chunk
    return generated,records,count
