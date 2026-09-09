#!/usr/bin/env python3
"""Validate actual complete outputs after constructor-only changes."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);result=dict(status='running')
    try:
        path=args.case/'summary.json';reference_path=args.reference/'summary.json'
        d=json.loads(path.read_text());ref=json.loads(reference_path.read_text())
        assert d['status']==ref['status']=='pass' and d['observer_noise_latent_RGB_equivalence']
        assert d['constructor_mode']=='strict_checkpoint_no_parameter_init'
        init=d['parameter_initialization_policy'];assert init['enabled'] and init['skipped_parameter_initializer_calls']>0
        assert not d['strict_generator_load']['missing_keys'] and not d['strict_generator_load']['unexpected_keys']
        for key in ('seed','latent_shape','noise_sha256','latent_sha256','prompts_per_block','native_shot_pin_events',
                    'assets_manifest_sha256','source_files_sha256','fixed_native_adaln_recipe','triton_version',
                    'native_positive_and_negative_KV_bytes'):
            assert d[key]==ref[key],key
        actual=torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True)
        reference=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
        assert list(actual.shape)==list(reference.shape)==d['latent_shape'] and torch.equal(actual,reference)
        if d.get('causal_scene_memory') is not None:
            assert d['causal_scene_memory']['archives']==ref['causal_scene_memory']['archives']
            assert d['causal_scene_memory']['decisions']==ref['causal_scene_memory']['decisions']
        hashes=[]
        for root in (args.case,args.reference):
            digest=hashlib.sha256();count=0
            with av.open(str(root/'video.mp4')) as video:
                video.streams.video[0].codec_context.thread_count=2
                for frame in video.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
            assert count==d['pixel_frames'];hashes.append(digest.hexdigest())
        assert hashes[0]==hashes[1]
        result.update(status='pass',actual_full_latent_and_decoded_RGB_exact=True,frames=d['pixel_frames'],
            summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_summary_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            skipped_parameter_initializer_calls=init['skipped_parameter_initializer_calls'],
            constructor_load_s=d['load_s'],reference_load_s=ref['load_s'],
            timing_is_single_gate_observation_not_repeated_speedup=True,new_independent_quality_sample=False,
            not_attention_or_steady_state_speedup=True)
    except Exception:result.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
