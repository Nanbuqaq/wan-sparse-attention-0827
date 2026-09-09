#!/usr/bin/env python3
"""Single-layer, same-ROI measured pack/pin/onload replay; not video speedup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_region_layout import region_layout_plan


def stats(values):
    values=torch.tensor(values,dtype=torch.float64)
    return dict(median=float(values.quantile(.5)),p95=float(values.quantile(.95)))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--region',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--host-pinned-budget-mib',type=int,default=2048)
    p.add_argument('--prepacked-control',action='store_true',help='add the fair fixed-route CPU packed-cache control')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real physically locked GPU required')
    summary=json.loads((args.capture/'summary.json').read_text());assert summary['status']=='pass' and summary['observer_noise_latent_RGB_equivalence']
    path=args.capture/'attention_teacher.pt'
    with path.open('rb') as handle:capture_sha=hashlib.file_digest(handle,'sha256').hexdigest()
    assert capture_sha==summary['attention_teacher']['sha256']
    region=json.loads(args.region.read_text());assert region['capture_sha256']==capture_sha
    selected=[i-7040 for i in region['partitions']['source_red_pile']]
    data=torch.load(path,map_location='cpu',weights_only=True)
    record=next(r for r in data['records'] if r['phase']==0 and r['layer']==0)
    source=torch.stack([record[k][0,7040:14080].clone() for k in ('k','v')])
    del record,data
    assert source.shape==(2,7040,24,128) and source.dtype==torch.bfloat16
    selected_index=torch.tensor(sorted(selected));expected=source.index_select(1,selected_index).cuda()
    torch.cuda.synchronize();layouts=['exact','block64','page256','spatial4','spatial8','frame'];entries={};build=[]
    pinned_bytes=0;max_staging_bytes=0;staging_shape_bytes=0;budget=args.host_pinned_budget_mib*1024**2
    for layout in layouts:
        plan=region_layout_plan(selected,layout);index=torch.tensor(plan['physical_to_logical'])
        identity=plan['physical_to_logical']==list(range(7040))
        began=time.perf_counter()
        if identity:archive=source
        else:
            archive=source.index_select(1,index.clamp_min(0));archive[:,index<0]=0
        build_ms=1000*(time.perf_counter()-began)
        required=archive.numel()*archive.element_size()
        max_staging_bytes=max(max_staging_bytes,plan['scheduled_tokens']*2*24*128*2)
        staging_shape_bytes+=plan['scheduled_tokens']*2*24*128*2
        if pinned_bytes+required+2*staging_shape_bytes>budget:raise ValueError('explicit pinned tensor/cache-shape budget exceeded')
        began=time.perf_counter();pinned_archive=archive.pin_memory();pin_ms=1000*(time.perf_counter()-began)
        pinned_bytes+=required
        build.append(dict(layout=layout,layout_build_ms=build_ms,full_archive_pin_ms=pin_ms,
            archive_bytes=required,raw_layout_identity_reuses_existing_storage=identity,construction_timing_single_observation=True))
        entries[layout]=dict(plan=plan,archive=archive,pinned=pinned_archive,
            pack_index=torch.tensor(plan['packed_physical_indices']),gather_index=torch.tensor(plan['gather_indices'],device='cuda'),
            destination=torch.empty((2,plan['scheduled_tokens'],24,128),device='cuda',dtype=torch.bfloat16))
    variants=[(layout,mode) for layout in layouts for mode in ('staged_fused','staged_separate','pre_pinned_runs')]
    fixed_roi_cache=None
    if args.prepacked_control:
        needed=expected.numel()*expected.element_size()
        if pinned_bytes+2*staging_shape_bytes+needed>budget:raise ValueError('fixed-route pinned cache exceeds explicit shape budget')
        began=time.perf_counter();packed_source=source.index_select(1,selected_index);pinned_selected=packed_source.pin_memory()
        fixed_roi_cache=dict(pack_pin_creation_ms=1000*(time.perf_counter()-began),pinned_bytes=needed,
            selected_coordinate_sha256=entries['exact']['plan']['logical_coordinate_sha256'],
            invalidated_on_coordinate_or_source_storage_change=True,creation_timing_single_observation=True)
        entries['exact']['pinned_selected']=pinned_selected
        variants.append(('exact','prepacked_exact'))
    results={variant:[] for variant in variants};rng=random.Random(20260909)
    def run_one(layout,mode):
        entry=entries[layout];plan=entry['plan'];destination=entry['destination']
        torch.cuda.synchronize();began=time.perf_counter()
        if mode=='staged_fused':
            packed=entry['archive'].index_select(1,entry['pack_index']);pinned=packed.pin_memory()
        elif mode=='staged_separate':
            packed=[entry['archive'][i].index_select(0,entry['pack_index']) for i in (0,1)]
            pinned=[x.pin_memory() for x in packed]
        elif mode=='prepacked_exact':pinned=entry['pinned_selected']
        else:pinned=entry['pinned']
        packed_at=time.perf_counter();copy_start=torch.cuda.Event(enable_timing=True);copy_end=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        copy_start.record()
        if mode in ('staged_fused','prepacked_exact'):destination.copy_(pinned,non_blocking=True);copies=1
        elif mode=='staged_separate':
            for i in (0,1):destination[i].copy_(pinned[i],non_blocking=True)
            copies=2
        else:
            copies=0
            for start,finish,target in plan['runs']:
                for i in (0,1):
                    part=pinned[i,start:finish];assert part.is_contiguous()
                    destination[i,target:target+finish-start].copy_(part,non_blocking=True);copies+=1
        identity_gather=plan['gather_indices']==list(range(plan['scheduled_tokens']))
        copy_end.record();actual=destination if identity_gather else destination.index_select(1,entry['gather_index'])
        end.record();end.synchronize()
        wall_ms=1000*(time.perf_counter()-began)
        # Validation is deliberately outside timed work, but executed on every replay.
        assert torch.equal(actual,expected)
        return dict(pack_pin_ms=1000*(packed_at-began),copy_stream_span_ms=copy_start.elapsed_time(copy_end),
            GPU_gather_span_ms=copy_end.elapsed_time(end),wall_ms=wall_ms,copy_calls=copies)
    # Five whole-grid warmup rounds, then 30 seeded shuffled measurement rounds.
    for repetition in range(35):
        order=variants.copy();rng.shuffle(order)
        for variant in order:
            result=run_one(*variant)
            if repetition>=5:results[variant].append(result)
    rows=[];bytes_per_token=2*24*128*2
    for (layout,mode),measurements in results.items():
        plan=entries[layout]['plan'];physical_bytes=plan['scheduled_tokens']*bytes_per_token
        rows.append(dict(layout=layout,mode=mode,logical_coordinate_sha256=plan['logical_coordinate_sha256'],
            logical_tokens=len(selected),scheduled_tokens=plan['scheduled_tokens'],H2D_bytes=physical_bytes,
            padding_bytes=plan['padding_tokens']*bytes_per_token,copy_calls=measurements[0]['copy_calls'],
            GPU_gather_elided=plan['gather_indices']==list(range(plan['scheduled_tokens'])),
            measurement_count=30,metrics={field:stats([r[field] for r in measurements]) for field in ('pack_pin_ms','copy_stream_span_ms','GPU_gather_span_ms','wall_ms')},
            full_sample_list=measurements,all_selected_KV_bitwise_equal=True))
    report=dict(status='pass',GPU=torch.cuda.get_device_name(),torch=torch.__version__,source_layer=0,source_phase=0,
        source_capture_sha256=capture_sha,region_sha256=hashlib.sha256(args.region.read_bytes()).hexdigest(),
        rows=rows,archive_construction=build,source_shape=list(source.shape),CPU_threads=2,warmup_rounds=5,measurement_rounds=30,
        fixed_route_packed_cache=fixed_roi_cache,
        CPU_affinity=sorted(os.sched_getaffinity(0)),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        pinned_archive_bytes_across_variants=pinned_bytes,max_transient_staging_pinned_bytes=max_staging_bytes,
        host_pinned_tensor_shape_budget_bytes=budget,
        conservative_requested_archive_plus_two_staging_sets_bytes=pinned_bytes+2*staging_shape_bytes+(fixed_roi_cache['pinned_bytes'] if fixed_roi_cache else 0),
        allocator_reserved_host_bytes_measured=False,
        compile_ms=None,no_custom_JIT_kernel=True,physical_padding_is_not_logical_attention=True,
        limitations=['single actual layer replay, not all30layer end-to-end video',
            'ROI is a past color diagnostic, not a causal online selector',
            'pre-pinned mode pays entire archive pin once; do not hide this residency cost',
            'host pinned budget covers requested tensor shapes, not measured allocator-reserved pages',
            'CUDA copy stream span can include launch gaps; not a hardware busy-time counter',
            'event allocation and launch overhead are included in wall time',
            'staged demand pack/pin and GPU gather counted; layout construction reported separately',
            'source capture/loading/extraction is setup, not measured archive D2H production cost',
            'prepacked_exact requires unchanged selected coordinates and source storage; not a free dynamic-route layout',
            'no overlap or Attention speedup claim'])
    (args.output/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(layout=r['layout'],mode=r['mode'],H2D_MB=r['H2D_bytes']/1e6,wall_ms=r['metrics']['wall_ms']) for r in rows],indent=2))


if __name__=='__main__':main()
