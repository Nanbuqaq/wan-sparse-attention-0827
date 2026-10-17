"""Atomic native-shot progress, RNG and generated-only history restoration."""
import hashlib,os
from pathlib import Path
import torch


def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()


def save_progress(path,*,case_sha256,checkpoint_sha256,weights_spec_sha256,case_id,mode,generated,records,generator):
 value=dict(schema='h3_native_progress_v1',case_sha256=case_sha256,checkpoint_sha256=checkpoint_sha256,weights_spec_sha256=weights_spec_sha256,
  case_id=case_id,mode=mode,completed_chunks=len(records),generated_clips=generated,records=records,
  generator_state=generator.get_state().cpu())
 path=Path(path);temp=path.with_name(path.name+f'.tmp.{os.getpid()}');torch.save(value,temp);os.replace(temp,path)


def restore(path,expected_sha256,*,case_sha256,checkpoint_sha256,weights_spec_sha256,case_id,mode,rows,episode,history,generator):
 if file_sha(path)!=expected_sha256:raise ValueError('Native progress checksum changed')
 value=torch.load(path,map_location='cpu',weights_only=True)
 if value.get('schema')!='h3_native_progress_v1' or any(value.get(k)!=v for k,v in dict(case_sha256=case_sha256,checkpoint_sha256=checkpoint_sha256,weights_spec_sha256=weights_spec_sha256,case_id=case_id,mode=mode).items()):
  raise ValueError('Native progress belongs to another case/checkpoint/history policy')
 count=value['completed_chunks'];records=value['records'];generated=value['generated_clips']
 if not 0<count<=len(rows) or len(records)!=count:raise ValueError('Invalid native committed prefix')
 written={}
 for number,(row,record) in enumerate(zip(rows[:count],records)):
  if record['chunk']!=number or record['row']!=row or record['state_mode']!=mode:raise ValueError('Native progress rows differ from episode')
  ci=row['clip_index'];a,b=row['clip_video_latent_start'],row['clip_video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end'];clip=episode['clips'][ci]
  if ci not in generated:raise ValueError('Native progress has missing clip')
  values=generated[ci]
  if values['video'].shape!=(24,clip['video_latent_count'],48,84) or values['audio'].shape!=(2,32,round(clip['num_frames']/24*40)):
   raise ValueError('Native progress changed clip geometry')
  if not torch.isfinite(values['video']).all() or not torch.isfinite(values['audio']).all():raise ValueError('Native progress is nonfinite')
  state=written.setdefault(ci,[0,0])
  if a!=state[0] or aa!=state[1]:raise ValueError('Native progress has uncommitted gaps')
  state[:]=[b,ae];evicted=history.commit(row,values['video'][:,a:b])
  if evicted!=record['evicted_frame_ids'] or history.bytes!=record['active_history_bytes']:raise ValueError('Native restored writeback/eviction differs')
 if set(generated)!=set(written):raise ValueError('Unobserved future clip in native progress')
 for ci,(end,aend) in written.items():
  if torch.count_nonzero(generated[ci]['video'][:,end:]) or torch.count_nonzero(generated[ci]['audio'][:,:,aend:]):
   raise ValueError('Uncommitted future values in native progress')
 generator.set_state(value['generator_state']);return generated,records,count
