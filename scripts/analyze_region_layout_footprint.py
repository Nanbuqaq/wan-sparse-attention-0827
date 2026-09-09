#!/usr/bin/env python3
"""Computed physical coverage of one fixed ROI; no GPU speed/IO measurement."""
import argparse
import hashlib
import json
from pathlib import Path


def footprint(ids,frames=8,height=22,width=40):
    selected=sorted(set(ids));frame_tokens=height*width;total=frames*frame_tokens
    if not selected or selected[0]<0 or selected[-1]>=total:raise ValueError('invalid ROI coordinates')
    def runs(values):return sum(i==0 or value!=values[i-1]+1 for i,value in enumerate(values))
    rows=[dict(layout='exact',payload_tokens=len(selected),scheduled_tokens=len(selected),physical_runs=runs(selected))]
    blocks=sorted({(token//frame_tokens,(token%frame_tokens)//64) for token in selected})
    rows.append(dict(layout='frame_linear_Block64',payload_tokens=sum(min(64,frame_tokens-b*64) for f,b in blocks),
                     scheduled_tokens=64*len(blocks),physical_runs=runs([f*((frame_tokens+63)//64)+b for f,b in blocks])))
    pages=sorted({i//256 for i in selected})
    rows.append(dict(layout='flat_Page256',payload_tokens=sum(min(256,total-p*256) for p in pages),
                     scheduled_tokens=256*len(pages),physical_runs=runs(pages)))
    for edge in (4,8):
        tiles=set()
        for t in selected:
            f,r=divmod(t,frame_tokens);y,x=divmod(r,width);tiles.add((f,y//edge,x//edge))
        th=(height+edge-1)//edge;tw=(width+edge-1)//edge
        rows.append(dict(layout=f'spatial_{edge}x{edge}',
            payload_tokens=sum(min(edge,height-y*edge)*min(edge,width-x*edge) for f,y,x in tiles),
            scheduled_tokens=edge*edge*len(tiles),physical_runs=runs(sorted(f*th*tw+y*tw+x for f,y,x in tiles))))
    touched_frames={i//frame_tokens for i in selected}
    rows.append(dict(layout='Frame880',payload_tokens=len(touched_frames)*frame_tokens,
                     scheduled_tokens=len(touched_frames)*frame_tokens,physical_runs=runs(sorted(touched_frames))))
    bytes_per_token=24*128*2*2*30
    for row in rows:
        row.update(logical_tokens=len(selected),computed_payload_bytes=row['payload_tokens']*bytes_per_token,
                   computed_padding_bytes=(row['scheduled_tokens']-row['payload_tokens'])*bytes_per_token,
                   payload_amplification=row['payload_tokens']/len(selected),
                   all_heads_all_layers_KV_copy_runs=row['physical_runs']*2*30)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.diagnostics.read_text());ids=[i-7040 for i in data['partitions']['source_red_pile']]
    result=dict(scope='computed coverage for fixed logical ROI; original coordinates and RoPE unchanged',
        actual_GPU_bytes_or_speed_measured=False,source_diagnostic_sha256=hashlib.sha256(args.diagnostics.read_bytes()).hexdigest(),
        limitations=['red color component can include falling beads','equal-token background was not spatial-distance matched',
                     'spatial layouts would require a new physical packing implementation','no packing or onload latency measured'],
        rows=footprint(ids))
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result['rows'],indent=2))


if __name__=='__main__':main()
