"""Restore only a verified committed prefix; never regenerate completed chunks."""
import copy
import hashlib
import json
from pathlib import Path

import torch


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def restore_prefix(task,case,source,spec,nfe):
    if case.get('chunk_policy')!='fixed5' or task['history_mode'] not in ['carry','zero']:
        raise ValueError('Prefix recovery supports fixed5 carry/zero only')
    path=Path(task['resume_snapshot']);receipt=Path(task['resume_records'])
    if file_sha(path)!=task['resume_snapshot_sha256'] or file_sha(receipt)!=task['resume_records_sha256']:
        raise ValueError('Frozen recovery snapshot/records changed')
    saved=torch.load(path,map_location='cpu',weights_only=True);meta=json.loads(receipt.read_text())
    first=int(task['resume_first_chunk']);last=first-1
    if not 0<first<len(case['chunks']) or saved['completed_NN_chunk']!=last or meta['first_chunk']!=first:
        raise ValueError('Recovery must start immediately after an unfinished committed prefix')
    if saved['case_id']!=case['case_id'] or saved['seed']!=case['seed'] or saved['nfe']!=nfe or saved['state_mode']!=task['history_mode']:
        raise ValueError('Prefix belongs to another case/seed/sampler/history control')
    if saved['adapter_mode']!=('native_init' if spec['kind']=='base' else 'trained'):
        raise ValueError('Prefix model kind differs from the requested session')
    original=json.loads(Path(meta['original_config_path']).read_text())
    if file_sha(meta['original_config_path'])!=meta['original_config_sha256']:
        raise ValueError('Archived source configuration changed')
    if original['case']!=case or original['source_contract']!=source:
        raise ValueError('Recovery changes the source-time or chunk contract')
    keys=['kind','checkpoint_sha256','full_native_precision','zero_lora_interface']
    if any(original['weights_spec'].get(k)!=spec.get(k) for k in keys):
        raise ValueError('Recovery changes model weights or architecture/precision')
    if original['actual_sampling_NFE']!=nfe or original['actual_CFG']!=1.:
        raise ValueError('Recovery changes NFE/CFG')
    row=case['chunks'][last];end=row['video_latent_end'];audio_end=row['audio_latent_end']
    if saved['current_chunk']!=row or saved['generated_ids']!=list(range(end)):
        raise ValueError('Prefix geometry or committed-frame lineage differs')
    video,audio=saved['video'],saved['audio']
    if video.shape!=(24,source['video_latent_count'],48,84) or audio.shape!=(2,32,source['audio_latent_count']):
        raise ValueError('Recovery archive shape differs')
    if video.dtype!=torch.bfloat16 or audio.dtype!=torch.bfloat16 or not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError('Invalid recovery tensors')
    if bool(video[:,end:].any()) or bool(audio[:,:,audio_end:].any()):
        raise ValueError('Uncommitted future values appear in the saved prefix')
    records=meta['records']
    if len(records)!=first or records[:last]!=saved['previous_complete_records']:
        raise ValueError('Committed prefix records are incomplete')
    for i,record in enumerate(records):
        wanted=case['chunks'][i]
        if record['chunk']!=i or record['window_id']!=wanted['window_id'] or record['nfe']!=nfe or record['cfg_scale']!=1.:
            raise ValueError('Prefix record changed its generation contract')
    for name in ['video','audio']:
        if tensor_sha(saved['initial_'+name+'_noise'])!=saved['current_'+name+'_noise_sha256'] or records[-1][name+'_noise_sha256']!=saved['current_'+name+'_noise_sha256']:
            raise ValueError('Last committed chunk noise evidence changed')
    if file_sha(row['prompt_path'])!=saved['current_prompt_sha256']:
        raise ValueError('Last committed prompt cache changed')
    state=saved['noise_generator_state']
    if state.dtype!=torch.uint8 or state.ndim!=1:raise ValueError('Invalid saved CUDA RNG state')
    reused=[]
    for record in records:
        item=copy.deepcopy(record);item['reused_from_resume_prefix']=True;item.setdefault('origin_job_id',meta['origin_job_id'])
        if item.get('chunk_snapshot') in meta.get('archived_snapshots',{}):
            item['origin_chunk_snapshot']=item['chunk_snapshot'];item['chunk_snapshot']=meta['archived_snapshots'][item['chunk_snapshot']]
        reused.append(item)
    return dict(first_chunk=first,video=video.clone(),audio=audio.clone(),generated_ids=set(saved['generated_ids']),
                records=reused,generator_state=state.clone(),
                provenance=dict(snapshot=str(path),snapshot_sha256=task['resume_snapshot_sha256'],
                                records=str(receipt),records_sha256=task['resume_records_sha256'],origin_job_id=meta['origin_job_id'],
                                committed_chunks=first,committed_video_latents=end,committed_audio_latents=audio_end,
                                generator_state_sha256=tensor_sha(state),reused_prefix_dit_seconds=sum(r['denoise_seconds'] for r in records)))
