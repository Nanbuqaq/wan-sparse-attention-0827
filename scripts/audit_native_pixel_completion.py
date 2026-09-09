#!/usr/bin/env python3
"""Actual output equality, byte accounting and encode-buffer ownership audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_pipeline_gate import reference_geometry


def validate_groups(stream):
    groups=[g for r in stream['records'] for g in r['groups']];position=0;last_release={};last_sink=0.
    for group in groups:
        assert group['start_pixel']==position
        slot=group['pixel_buffer_slot'];assert 0<=slot<stream['pixel_buffer_slots']
        assert group['pixel_buffer_acquired_s']>=last_release.get(slot,0.)
        assert group['sink_finished_s']>=group['CPU_pixels_ready_s']>=group['pixel_buffer_acquired_s']
        assert group['sink_finished_s']>=last_sink;last_sink=group['sink_finished_s']
        last_release[slot]=group['sink_finished_s'];position+=group['pixel_frames']
    assert position==stream['completed_pixel_frames']==stream['encoded_pixels']
    assert all(r['finished_s']>=max(g['sink_finished_s'] for g in r['groups']) for r in stream['records'])
    return groups


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    ref=json.loads((args.reference/'summary.json').read_text());assert ref['status']=='pass';geometry=reference_geometry(ref)
    reference=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
    def video_hash(root):
        digest=hashlib.sha256();count=0
        with av.open(str(root/'video.mp4')) as video:
            video.streams.video[0].codec_context.thread_count=2
            for frame in video.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
        return count,digest.hexdigest()
    reference_video=video_hash(args.reference);rows=[]
    for mode in ('inline','thread'):
        root=args.root/mode;row=dict(mode=mode)
        try:
            path=root/'summary.json';d=json.loads(path.read_text())
            assert d['status']=='pass' and d['observer_noise_latent_RGB_equivalence']
            for key in ('seed','noise_sha256','latent_sha256','latent_shape','prompts_per_block','native_shot_pin_events',
                        'fixed_native_adaln_recipe','triton_version','assets_manifest_sha256','source_files_sha256'):
                assert d[key]==ref[key],key
            assert d['physical_GPU_mapping']=='0,1'
            stream=d['video_pipeline'];assert stream['encode_mode']==mode and stream['scheduling']=='overlap_two_gpu'
            assert stream['streamed_latent_sha256']==d['latent_sha256']
            assert torch.equal(torch.load(root/'latents.pt',map_location='cpu',weights_only=True),reference)
            assert video_hash(root)==reference_video and reference_video[0]==geometry['pixel_frames']
            assert stream['latent_D2H_bytes']==stream['latent_H2D_bytes']==geometry['latent_bytes']
            assert stream['pixel_D2H_bytes']==geometry['pixel_D2H_bytes']
            assert stream['pinned_input_bytes']+stream['pinned_output_bytes']<=stream['pinned_budget_bytes']
            assert stream['pixel_buffer_slots']==(1 if mode=='inline' else 2)
            assert [r['start_latent'] for r in stream['records']]==geometry['chunk_starts']
            groups=validate_groups(stream);assert len(groups)==geometry['latent_frames']
            trace=json.loads((root/'pipeline_host_trace.json').read_text())['traceEvents']
            enc={e['tid'] for e in trace if e['name']=='encode'};dec={e['tid'] for e in trace if e['name']=='VAE_decode_group'}
            assert len(enc)==len(dec)==1
            assert (enc==dec) if mode=='inline' else enc.isdisjoint(dec)
            row.update(status='pass',actual_full_latent_and_decoded_RGB_exact=True,frames=geometry['pixel_frames'],
                pixel_slots_not_reused_before_sink_completion=True,decode_and_encode_threads_distinct=enc.isdisjoint(dec),
                complete_s=stream['complete_s'],first_packet_s=d['pixels']['first_packet_muxed_s'],
                producer_backpressure_s=stream['producer_backpressure_s'],pixel_backpressure_s=stream['pixel_output_backpressure_s'],
                pinned_input_bytes=stream['pinned_input_bytes'],pinned_output_bytes=stream['pinned_output_bytes'],
                GPU0_peak_allocated_bytes=d['generation_peak_allocated_bytes'],GPU1_peak_allocated_bytes=d['pipeline_VAE_GPU_peak_allocated_bytes'],
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception:row.update(status='fail',traceback=traceback.format_exc())
        rows.append(row)
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',rows=rows,
        GPU_overlap_requires_CUPTI_not_proven_by_CPU_threads=True,single_pair_not_repeated_speedup=True,
        numerical_gate_not_new_quality_sample=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
