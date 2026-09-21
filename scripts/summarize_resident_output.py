"""CPU-only resident paired timing collector. No performance claim without all lanes."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(root, pairs=4, lanes=4):
    batches=[];chunks=[];identity={}
    for pair in range(pairs):
        for mode in ('inline','thread'):
            cases=[]
            for lane in range(lanes):
                path=root/f'lane{lane}'/f'pair{pair}_{mode}'/'summary.json'
                d=json.loads(path.read_text());v=d['video_pipeline']
                assert d['status']=='pass' and d['measured'] is True
                assert d['case']==dict(id=f'pair{pair}_{mode}',mode=mode,measured=True,pair=pair)
                assert d['pixels']['frames']==v['encoded_pixels']==509
                assert v['encode_mode']==mode and v['scheduling']=='overlap_two_gpu'
                assert d['full_latent_equal_to_first_warmup']
                keys=('runner_commit','assets_manifest_sha256','seed','latent_sha256','physical_GPU_mapping','gpu')
                signature=tuple(d[k] for k in keys)+(d['pixels']['raw_RGB_sha256'],)
                if lane in identity:assert identity[lane]==signature,'case identity or output changed'
                else:identity[lane]=signature
                assert d['monotonic_delivery_ns']>d['monotonic_start_ns']
                cases.append(d)
                for r in v['records']:
                    chunks.append(dict(pair=pair,mode=mode,lane=lane,start_latent=r['start_latent'],
                        delivered_s=r['finished_s'],submitted_s=r['submitted_s'],
                        submission_to_delivery_s=r['finished_s']-r['submitted_s']))
            starts=[d['monotonic_start_ns'] for d in cases]
            ends=[d['monotonic_delivery_ns'] for d in cases]
            wall=(max(ends)-min(starts))/1e9
            common=(min(ends)-max(starts))/1e9
            batches.append(dict(pair=pair,mode=mode,batch_delivery_s=wall,
                aggregate_delivered_fps=509*lanes/wall,start_skew_s=(max(starts)-min(starts))/1e9,
                common_active_window_s=max(0,common),
                per_lane_complete_s=[d['video_pipeline']['complete_s'] for d in cases],
                per_lane_first_packet_s=[d['pixels']['first_packet_muxed_s'] for d in cases]))
    # Structural validation is distinct from actual MP4 decoding and timing validity.
    paired=[]
    for pair in range(pairs):
        a,b=[next(x for x in batches if x['pair']==pair and x['mode']==m) for m in ('inline','thread')]
        paired.append(dict(pair=pair,delivery_delta_s=b['batch_delivery_s']-a['batch_delivery_s'],
            delivery_reduction_fraction=1-b['batch_delivery_s']/a['batch_delivery_s']))
    return dict(status='structural_checks_pass',batches=batches,pairs=paired,
                median_pair_reduction_fraction=statistics.median(x['delivery_reduction_fraction'] for x in paired),
                MP4_decode_audit='required_separately',timing_validity='review_start_skew_and_resource_evidence'),chunks


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:
        report,chunks=summarize(a.root)
        with (a.output/'chunks.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(chunks[0]));w.writeheader();w.writerows(chunks)
    except Exception as e:
        (a.output/'summary.json').write_text(json.dumps(dict(status='fail',error=repr(e)),indent=2));raise
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
