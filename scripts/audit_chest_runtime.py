#!/usr/bin/env python3
"""First full-size registered chest runtime gate; no source-choice oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.review_chest_causal_memory import audit_memory


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);result=dict(status='running')
    try:
        path=args.case/'summary.json';reference_path=args.reference/'summary.json'
        d=json.loads(path.read_text());ref=json.loads(reference_path.read_text())
        assert d['status']==ref['status']=='pass' and d['seed']==ref['seed']==20260925
        assert d['latent_shape']==ref['latent_shape']==[1,128,48,44,80]
        for key in ('noise_sha256','pre_return_latent_sha256','prompts_per_block','source_files_sha256',
                    'assets_manifest_sha256','fixed_native_adaln_recipe','triton_version','native_shot_pin_events'):
            assert d[key]==ref[key],key
        protocol=d['object_state_protocol']
        registration=ROOT/'configs/system/native_object_state_memory.json'
        assert not protocol['Dense_only'] and not protocol['formal_holdout']
        assert protocol['memory_registration_sha256']==hashlib.sha256(registration.read_bytes()).hexdigest()
        details=audit_memory(d['causal_scene_memory'],'original')
        actual=torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True)
        reference=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
        assert list(actual.shape)==list(reference.shape)==d['latent_shape'] and torch.isfinite(actual).all()
        assert torch.equal(actual[:,:96],reference[:,:96])
        hashes=[]
        for root in (args.case,args.reference):
            digest=hashlib.sha256();n=0
            with av.open(str(root/'video.mp4')) as video:
                video.streams.video[0].codec_context.thread_count=2
                for i,frame in enumerate(video.decode(video=0)):
                    if i<381:digest.update(frame.to_ndarray(format='rgb24').tobytes())
                    n+=1
            assert n==509;hashes.append(digest.hexdigest())
        assert hashes[0]==hashes[1]
        result.update(status='pass',actual_pre96_and_pre381_decoded_RGB_exact=True,details=details,
            full_video_technical_gate_not_quality_success=True,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reference_summary_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
    except Exception:result.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
