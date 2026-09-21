#!/usr/bin/env python3
import argparse, hashlib, json, os, platform
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--mode',required=True); p.add_argument('--exit-status',type=int,required=True); a=p.parse_args()
rows=[]
for lane in range(4):
    path=a.root/f'lane{lane}'/'summary.json'
    if not path.exists(): rows.append(dict(lane=lane,status='missing')); continue
    d=json.loads(path.read_text()); vp=d.get('video_pipeline',{}); px=d.get('pixels',{})
    rows.append(dict(lane=lane,status=d.get('status'),physical_GPU_mapping=d.get('physical_GPU_mapping'),latent_sha256=d.get('latent_sha256'),rgb_sha256=px.get('raw_RGB_sha256'),frames=px.get('frames'),complete_s=vp.get('complete_s'),first_packet_s=px.get('first_packet_muxed_s'),first_rgb_s=px.get('first_RGB_ready_s'),producer_backpressure_s=vp.get('producer_backpressure_s'),pixel_output_backpressure_s=vp.get('pixel_output_backpressure_s'),GPU0_kernel_s=d.get('GPU0_kernel_s'),GPU1_kernel_s=d.get('GPU1_kernel_s'),summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
out=dict(status='pass' if a.exit_status==0 and len(rows)==4 and all(r['status']=='pass' for r in rows) else 'fail',mode=a.mode,host=platform.node(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),lanes=rows,exit_status=a.exit_status)
(a.root/'summary_8gpu.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); raise SystemExit(0 if out['status']=='pass' else 1)
