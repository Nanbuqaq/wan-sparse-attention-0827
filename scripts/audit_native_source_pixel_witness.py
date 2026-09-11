#!/usr/bin/env python3
"""CPU audit of live raw source windows, complete outputs, and codec differences."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    d=json.loads((args.case/'summary.json').read_text());ref=json.loads((args.reference/'summary.json').read_text())
    if d['status']!='pass' or ref['status']!='pass':raise ValueError('successful capture and reference required')
    latent=torch.load(args.case/'latents.pt',weights_only=True,map_location='cpu')
    old=torch.load(args.reference/'latents.pt',weights_only=True,map_location='cpu')
    if not torch.equal(latent,old) or tensor_sha256(latent)!=d['latent_sha256']:raise ValueError('actual complete latent differs')
    del old
    path=args.case/'source_raw_rgb.pt';data=torch.load(path,weights_only=True,map_location='cpu')
    if not data['complete'] or not data['pixels_before_lossy_codec']:raise ValueError('incomplete raw source witness')
    archives={r['archive_version']:r for r in d['causal_block_memory']['scene_selection']['archives']}
    if len(data['records'])!=len(archives):raise ValueError('source archive count differs')
    rows=[]
    for r in data['records']:
        start,end=r['source_start'],r['source_end'];a,b=max(0,4*start-3),4*end-3
        if archives[r['archive_version']]['source_end']!=end or r['pixel_start']!=a or r['pixel_end']!=b:
            raise ValueError('source event/pixel bounds differ')
        if tensor_sha256(latent[:,start:end])!=r['source_latent_sha256']:raise ValueError('source latent ownership hash differs')
        pixels=r['pixels']
        if pixels.dtype!=torch.uint8 or list(pixels.shape)!=[b-a,latent.shape[-2]*16,latent.shape[-1]*16,3]:
            raise ValueError('raw source RGB shape differs')
        if hashlib.sha256(memoryview(pixels.numpy())).hexdigest()!=r['raw_pixel_bytes_sha256']:
            raise ValueError('actual raw source hash differs')
        rows.append({k:v for k,v in r.items() if k!='pixels'}|dict(codec_squared_error=0,codec_changed_values=0,
            codec_max_abs=0,codec_values=0,decoded_source_frames=0))
    del latent
    digests={};counts={}
    for name,root in (('capture',args.case),('reference',args.reference)):
        digest=hashlib.sha256();count=0
        with av.open(str(root/'video.mp4')) as c:
            c.streams.video[0].codec_context.thread_count=2
            for frame in c.decode(video=0):
                rgb=frame.to_ndarray(format='rgb24');digest.update(memoryview(rgb))
                if name=='capture':
                    for row,source in zip(rows,data['records']):
                        if row['pixel_start']<=count<row['pixel_end']:
                            raw=source['pixels'][count-row['pixel_start']].numpy()
                            diff=rgb.astype(np.int16)-raw.astype(np.int16)
                            row['codec_squared_error']+=int(np.square(diff.astype(np.int32)).sum(dtype=np.int64))
                            row['codec_changed_values']+=int(np.count_nonzero(diff))
                            row['codec_max_abs']=max(row['codec_max_abs'],int(np.abs(diff).max()))
                            row['codec_values']+=diff.size;row['decoded_source_frames']+=1
                count+=1
        digests[name]=digest.hexdigest();counts[name]=count
    if digests['capture']!=digests['reference'] or counts['capture']!=counts['reference'] or counts['capture']!=d['pixels']['frames']:
        raise ValueError('actual decoded complete video differs')
    for row in rows:
        if row['decoded_source_frames']!=row['pixel_end']-row['pixel_start']:raise ValueError('source video decode incomplete')
        row['codec_MSE_uint8']=row['codec_squared_error']/row['codec_values']
        row['codec_changed_fraction']=row['codec_changed_values']/row['codec_values']
    with path.open('rb') as f:raw_sha=hashlib.file_digest(f,'sha256').hexdigest()
    witness=d['source_pixel_witness']
    report=dict(status='pass',actual_complete_latent_and_decoded_RGB_exact=True,decoded_frames=counts['capture'],
        raw_payload_sha256=raw_sha,windows=rows,witness=witness,
        extra_latent_hash_CPU_s=sum(r.get('witness_hash_CPU_s',0) for r in d['video_pipeline']['records']),
        extra_latent_hash_CPU_bytes=sum(r.get('witness_hash_CPU_bytes',0) for r in d['video_pipeline']['records']),
        limitations=['raw RGB differs from decoded lossy video; neither input may silently replace the other',
            'bounded observation only, not online segmentation/selection or an arbitrary-duration archive manager'])
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',windows=len(rows),decoded_frames=counts['capture'],
        codec_MSE=[r['codec_MSE_uint8'] for r in rows],CPU_owned_peak_bytes=witness['CPU_owned_peak_bytes'])))


if __name__=='__main__':main()
