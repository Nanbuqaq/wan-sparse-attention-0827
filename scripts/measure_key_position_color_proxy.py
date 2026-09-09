#!/usr/bin/env python3
"""Supplementary color occupancy, never bead count, state fidelity, or physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def normalize_occupancy(source,values):
    reference=float(np.median(source))
    if reference<=1e-6:raise ValueError('source has no measurable red occupancy')
    return reference,np.asarray(values,dtype=float)/reference


def main():
    import av
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.analyze_native_state_region import red_cells
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    groups=[];sources=[];fig,axes=plt.subplots(len(args.review),1,figsize=(10,3.4*len(args.review)),squeeze=False)
    colors={'related_original':'#555555','related_recent':'#bb2233','away_original':'#559944','away_recent':'#3366aa'}
    source_indices=list(range(160,189,4));return_indices=list(range(384,509,4))
    for group_index,review in enumerate(args.review):
        audit_path=review/'technical_audit.json';audit=json.loads(audit_path.read_text());rows=[];source_witness=None
        assert not audit['available_snapshot'] and len(audit['cases'])==4
        sources.append(dict(path=str(audit_path),sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest()))
        for row in audit['cases']:
            assert row['status']=='technical_pass' and row['pre96_latent_and_decoded_prefix_exact']
            summary_path=Path(row['source_summary']);assert hashlib.sha256(summary_path.read_bytes()).hexdigest()==row['summary_sha256']
            summary=json.loads(summary_path.read_text());video=summary_path.parent/'video.mp4';values={}
            with av.open(str(video)) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    if i in source_indices or i in return_indices:
                        rgb=frame.reformat(width=640,height=352).to_ndarray(format='rgb24')
                        fraction,_,_=red_cells(rgb,grid=(11,20));values[i]=float(fraction.mean())
            assert set(values)==set(source_indices+return_indices)
            source=np.array([values[i] for i in source_indices])
            if source_witness is None:source_witness=source
            assert np.array_equal(source,source_witness)
            target=np.array([values[i] for i in return_indices]);reference,ratio=normalize_occupancy(source,target)
            entry=dict(case=row['case'],seed=summary['seed'],summary_sha256=row['summary_sha256'],
                source_reference_red_fraction_median=reference,return_red_fraction=target.tolist(),
                return_relative_red_occupancy=ratio.tolist(),return_relative_occupancy_median=float(np.median(ratio)),
                first_chunk_relative_occupancy_median=float(np.median(ratio[:8])),late16_relative_occupancy_median=float(np.median(ratio[-16:])))
            rows.append(entry)
            axes[group_index,0].plot(return_indices,ratio,label=row['case'],color=colors[row['case']])
        axes[group_index,0].axhline(1,color='#999999',linestyle=':',linewidth=1,label='own source occupancy')
        axes[group_index,0].set(title=f'Seed {rows[0]["seed"]}: whole-frame red-color proxy',xlabel='Pixel frame',ylabel='Red area / own source median')
        axes[group_index,0].grid(alpha=.2);axes[group_index,0].legend(fontsize=8,ncol=3)
        groups.append(dict(seed=rows[0]['seed'],cases=rows))
    report=dict(schema='key_position_color_occupancy_v1',groups=groups,sources=sources,
        sampling=dict(source_pixel_indices=source_indices,return_pixel_indices=return_indices,analysis_resolution=[352,640]),
        threshold='same pre-existing red_cells HSV predicate: hue<=15 or>=330, saturation>=0.5, value>=0.2',
        quality_score=False,limitations=['full-frame color can include bottles or hands', 'not bead count or fill volume',
            'resizing and camera/pose affect occupancy','does not evaluate motion or material fidelity','descriptive corroboration only; not used to select policies'])
    (args.output/'color_occupancy.json').write_text(json.dumps(report,indent=2)+'\n')
    fig.tight_layout();fig.savefig(args.output/'color_occupancy.png',dpi=150);plt.close(fig)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>位置绑定颜色代理</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}</style><h1>补充描述：红色信息是否出现</h1><p>固定既有HSV阈值，按各seed自己的source面积归一化。不是珠数、填充高度、材质或物理质量，不能抵消视频中已观察到的额外动作。未用此结果调参。</p><img src="color_occupancy.png"><p><a href="color_occupancy.json">全部采样值、阈值、来源SHA</a></p>')
    print(json.dumps([dict(seed=g['seed'],cases={r['case']:r['return_relative_occupancy_median'] for r in g['cases']}) for g in groups]))


if __name__=='__main__':main()
