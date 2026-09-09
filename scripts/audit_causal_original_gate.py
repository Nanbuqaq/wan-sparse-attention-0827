#!/usr/bin/env python3
"""Actual-tensor/video equality for separately registered causal position controls."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def validate_memory_record(memory, policy, reference):
    assert policy in ('original','recent_virtual')
    assert memory['position_policy']==policy
    assert len(memory['installations'])==1
    entry=memory['installations'][0]
    plan=entry['installation']['admission_plan']
    assert entry['at_latent']==48 and plan['source_frames']==list(range(8,16))
    assert plan['temporal_delta']==(0 if policy=='original' else 48)
    assert memory['ledger']['history_H2D_KV_bytes']==377487360
    assert not memory['raw_source_frames_or_target_frames_supplied_to_selector']
    assert not memory['future_text_or_generated_outputs_read_by_selector']
    if policy=='recent_virtual':
        assert memory['archives']==reference['archives']
        assert memory['decisions']==reference['decisions']
        for key in ('archive_D2H_KV_bytes','history_H2D_KV_bytes','condition_summary_D2H_bytes',
                    'CPU_archive_peak_tensor_bytes','evicted_archives'):
            assert memory['ledger'][key]==reference['ledger'][key]
    return plan


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--position-policy',choices=('original','recent_virtual'),default='original')
    args=p.parse_args();torch.set_num_threads(2)
    result=dict(status='running')
    try:
        d=json.loads((args.case/'summary.json').read_text());ref=json.loads((args.reference/'summary.json').read_text())
        assert d['status']==ref['status']=='pass' and d['observer_noise_latent_RGB_equivalence']
        assert d['latent_shape']==ref['latent_shape']==[1,64,48,16,32]
        assert d['seed']==ref['seed']==20260904
        assert torch.equal(torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True),
                           torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True))
        def video_hash(path):
            digest=hashlib.sha256();n=0
            with av.open(str(path/'video.mp4')) as video:
                video.streams.video[0].codec_context.thread_count=2
                for frame in video.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());n+=1
            return n,digest.hexdigest()
        assert video_hash(args.case)==video_hash(args.reference)
        plan=validate_memory_record(d['causal_scene_memory'],args.position_policy,ref.get('causal_scene_memory'))
        result.update(status='pass',actual_full_latent_and_decoded_RGB_exact=True,position_policy=args.position_policy,
                      selected_source_frames=plan['source_frames'],source_selection_expectation_is_offline_only=True,
                      reference_summary_sha256=hashlib.sha256((args.reference/'summary.json').read_bytes()).hexdigest(),
                      summary_sha256=hashlib.sha256((args.case/'summary.json').read_bytes()).hexdigest())
    except Exception:result.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
