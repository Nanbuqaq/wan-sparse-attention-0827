"""Generated-only native prefixes and row-preserving content interventions."""
import hashlib
import json
from pathlib import Path
import torch
from h3_native_shot_contract import load_case,native_rows,NativeHistory

POLICIES=('full','original_content','recent_content','wrong_original')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_inputs(folder):
    folder=Path(folder);meta=json.loads((folder/'INPUT.json').read_text())
    for name in ['PREFIX.pt','NOISE.pt','DONOR_ORIGINAL.pt']:
        if file_sha(folder/name)!=meta['files_sha256'][name]:raise ValueError('Frozen probe input changed: '+name)
    if file_sha(meta['case_file'])!=meta['case_sha256']:raise ValueError('Native case changed')
    case,episode=load_case(meta['case_file']);rows=native_rows(episode);first=meta['first_replayed_chunk']
    if not 0<first<len(rows):raise ValueError('Invalid first native target')
    row=rows[first]
    if row!=meta['target_row'] or row['clip_video_latent_start']!=0 or row['clip_index']!=2:
        raise ValueError('Probe requires the original A-B-return first chunk')
    prefix=torch.load(folder/'PREFIX.pt',map_location='cpu',weights_only=True)
    if (prefix['completed_chunks']!=first or prefix['records']!=meta['prefix_records'] or
        len(prefix['records'])!=first or set(prefix['generated_clips'])!={0,1}):
        raise ValueError('Prefix includes missing or unobserved future clips')
    history=NativeHistory(episode);ends={}
    for number,(past,record) in enumerate(zip(rows[:first],prefix['records'])):
        if record['row']!=past or record['chunk']!=number or record['state_mode']!='carry':
            raise ValueError('Native prefix lineage changed')
        ci=past['clip_index'];values=prefix['generated_clips'][ci];clip=episode['clips'][ci]
        if values['video'].shape!=(24,clip['video_latent_count'],48,84) or values['audio'].shape!=(2,32,round(clip['num_frames']/24*40)):
            raise ValueError('Native prefix geometry changed')
        if not all(torch.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite prefix')
        a,b=past['clip_video_latent_start'],past['clip_video_latent_end'];aa,ae=past['audio_latent_start'],past['audio_latent_end']
        if [a,aa]!=ends.get(ci,[0,0]):raise ValueError('Uncommitted gaps in prefix')
        ends[ci]=[b,ae];evicted=history.commit(past,values['video'][:,a:b])
        if evicted!=record['evicted_frame_ids'] or history.bytes!=record['active_history_bytes']:
            raise ValueError('Prefix writeback/eviction differs')
    for ci,(end,aend) in ends.items():
        if torch.count_nonzero(prefix['generated_clips'][ci]['video'][:,end:]) or torch.count_nonzero(prefix['generated_clips'][ci]['audio'][:,:,aend:]):
            raise ValueError('Uncommitted prefix tail is nonzero')
    history.select(row)
    if set(meta['original_ids'])&set(meta['recent_ids']) or row['history_frame_ids']!=sorted(meta['original_ids']+meta['recent_ids']):
        raise ValueError('Original/recent groups must cover exact retained state')
    noise=torch.load(folder/'NOISE.pt',map_location='cpu',weights_only=True)
    if noise['row']!=row:raise ValueError('Noise belongs to a different target')
    for kind in ['video','audio']:
        if tensor_sha(noise['initial_'+kind+'_noise'])!=meta['target_record'][kind+'_noise_sha256']:
            raise ValueError('Actual target noise changed')
    donor=torch.load(folder/'DONOR_ORIGINAL.pt',map_location='cpu',weights_only=True)
    if donor['rows'].shape!=(1,len(meta['original_ids'])*1008,96) or donor['rows'].dtype!=torch.float32:
        raise ValueError('Donor does not preserve original-state row geometry')
    if not torch.isfinite(donor['rows']).all() or not donor['generated_past_only']:
        raise ValueError('Donor is not finite committed generated past')
    return meta,case,episode,history,noise,donor


def intervene_raw(history_rows,ids,original_ids,donor_rows,policy):
    if policy not in POLICIES:raise ValueError('Unknown native history intervention')
    if history_rows.shape!=(1,len(ids)*1008,96):raise ValueError('Actual native state row geometry changed')
    if policy!='wrong_original':return history_rows
    selected=torch.tensor([i in set(original_ids) for i in ids],dtype=torch.bool,device=history_rows.device).repeat_interleave(1008)
    output=history_rows.clone();output[:,selected]=donor_rows.to(output.device)
    if not torch.equal(output[:,~selected],history_rows[:,~selected]):raise ValueError('Wrong-original control changed recent past')
    return output
