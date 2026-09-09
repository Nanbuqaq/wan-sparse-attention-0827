#!/usr/bin/env python3
"""Actual files/traffic/ordering gate; CPU trace is not CUPTI overlap evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import math


def reference_geometry(reference):
    """Derive audit expectations from the actual paired native reference."""
    shape=reference['latent_shape']
    if (len(shape)!=5 or shape[0]!=1 or shape[2]!=48
            or any(type(x) is not int or x<=0 for x in shape) or shape[1]%8):
        raise ValueError('qualified batch1 native shape and eight-latent chunks required')
    frames=4*shape[1]-3
    if reference.get('pixel_frames')!=frames:
        raise ValueError('reference latent/pixel geometry disagrees')
    height,width=shape[3]*16,shape[4]*16
    return dict(shape=shape,latent_frames=shape[1],pixel_frames=frames,
                height=height,width=width,latent_bytes=math.prod(shape)*2,
                pixel_D2H_bytes=frames*3*height*width*4,
                chunk_starts=list(range(0,shape[1],8)))


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--available-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);ref=json.loads((args.reference/'summary.json').read_text());geometry=reference_geometry(ref)
    original=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
    def decoded_hash(root):
        digest=hashlib.sha256();count=0
        with av.open(str(root/'video.mp4')) as container:
            for frame in container.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
        return count,digest.hexdigest()
    ref_count,ref_rgb=decoded_hash(args.reference);rows=[]
    for mode in ('serial','overlap'):
        root=args.root/mode;path=root/'summary.json'
        if not path.exists():rows.append(dict(mode=mode,status='pending' if args.available_only else 'missing'));continue
        try:
            d=json.loads(path.read_text());assert d['status']=='pass' and d['observer_noise_latent_RGB_equivalence']
            for key in ('noise_sha256','latent_sha256','latent_shape','prompts_per_block','native_shot_pin_events','fixed_native_adaln_recipe'):
                assert d[key]==ref[key],key
            assert d['latent_shape']==geometry['shape'] and d['physical_GPU_mapping']=='0,1'
            stream=d['video_pipeline'];assert stream['scheduling']==mode+'_two_gpu'
            assert stream['streamed_latent_sha256']==d['latent_sha256']
            assert torch.equal(torch.load(root/'latents.pt',map_location='cpu',weights_only=True),original)
            assert decoded_hash(root)==(ref_count,ref_rgb) and ref_count==geometry['pixel_frames']
            assert stream['latent_D2H_bytes']==stream['latent_H2D_bytes']==geometry['latent_bytes']
            assert stream['pixel_D2H_bytes']==geometry['pixel_D2H_bytes']
            assert stream['pinned_input_bytes']+stream['pinned_output_bytes']<=stream['pinned_budget_bytes']
            assert [r['start_latent'] for r in stream['records']]==geometry['chunk_starts']
            groups=[g for r in stream['records'] for g in r['groups']]
            position=0
            for g in groups:assert g['start_pixel']==position;position+=g['pixel_frames']
            assert position==geometry['pixel_frames'] and len(groups)==geometry['latent_frames']
            trace=json.loads((root/'pipeline_host_trace.json').read_text())
            assert trace['metadata']['scope'].startswith('CPU host spans only')
            assert len({e['tid'] for e in trace['traceEvents']})>=2
            rows.append(dict(mode=mode,status='pass',actual_full_latent_and_decoded_RGB_exact=True,
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),streamed_latent_sha256=stream['streamed_latent_sha256'],
                completed_pixels=position,complete_s=stream['complete_s'],generation_host_s=stream['generation_host_s'],
                first_pixels_s=d['pixels']['first_RGB_ready_s'],first_packet_s=d['pixels']['first_packet_muxed_s'],
                producer_backpressure_s=stream['producer_backpressure_s'],
                GPU0_peak_allocated_bytes=d['generation_peak_allocated_bytes'],GPU1_peak_allocated_bytes=d['pipeline_VAE_GPU_peak_allocated_bytes'],
                pinned_input_bytes=stream['pinned_input_bytes'],pinned_output_bytes=stream['pinned_output_bytes']))
        except Exception:rows.append(dict(mode=mode,status='fail',traceback=traceback.format_exc()))
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'partial' if args.available_only and not any(r['status']=='fail' for r in rows) else 'fail',
        cases=rows,reference_geometry=geometry,no_new_independent_quality_samples=True,
        correctness_gate_not_repeated_speedup_evidence=True,
        small_gate_not_end_to_end_promotion=geometry['height']<704,GPU_overlap_proven=False)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
