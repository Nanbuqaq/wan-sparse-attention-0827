#!/usr/bin/env python3
"""Frozen paired native visual controls; only GPU work occupies inference lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--specs',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    specs=args.specs
    tasks=[('pattern',20261101),('close',20261101),('motion',20261101),('pattern_second',20261102)]
    plan=dict(schema='visual_restart_batch_v1',tasks=tasks,GPUs=4,lanes=2,
        modes=['t2v','i2v'],prefix='fixed past RGB from frozen artifacts',
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for label,_ in tasks
                      for p in (specs/(label+'.json'),specs/(label+'.png'))})
    if not args.run:
        print(json.dumps(plan,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'batch_plan.json').write_text(json.dumps(plan,indent=2))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=4 or len(set(visible))!=4:
        raise RuntimeError('exactly four worker-assigned GPUs required')
    import torch
    names=[torch.cuda.get_device_name(i) for i in range(4)]
    if len(set(names))!=1 or not any(x in names[0] for x in ('H200','H800')):
        raise RuntimeError('homogeneous H200 or H800 cohort required')
    (args.output/'hardware.json').write_text(json.dumps(dict(names=names,visible=visible,torch=torch.__version__),indent=2))
    def lane(index):
        pair=','.join(visible[2*index:2*index+2]);rows=[]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=pair,WAN_SPARSE_PHYSICAL_GPUS=pair)
        for label,seed in tasks[index::2]:
            pair_rows=[]
            for mode in ('t2v','i2v'):
                case=f'{label}_s{seed}_{mode}';out=args.output/case
                cmd=[sys.executable,str(ROOT/'scripts/run_native_visual_restart.py'),
                    '--assets',str(args.assets),'--source',str(ROOT/'third_party/LongLive2'),
                    '--spec',str(specs/(label+'.json')),'--output',str(out),
                    '--mode',mode,'--seed',str(seed)]
                began=time.time()
                with (args.output/(case+'.log')).open('x') as log:
                    rc=subprocess.call(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                row=dict(case=case,mode=mode,task=label,pair=pair,returncode=rc,command=cmd,elapsed_s=time.time()-began)
                if not rc:
                    summary=json.loads((out/'summary.json').read_text())
                    row.update(status=summary['status'],noise=summary['noise_sha256'],
                        pairs=summary['actual_self_attention_pairs_including_clean'],
                        artifacts={n:hashlib.sha256((out/n).read_bytes()).hexdigest()
                                   for n in ('summary.json','video.mp4','latents.pt')})
                (args.output/(case+'_terminal.json')).write_text(json.dumps(row,indent=2))
                rows.append(row);pair_rows.append(row)
                print(json.dumps(row),flush=True)
            if all(r['returncode']==0 for r in pair_rows):
                if len({r['noise'] for r in pair_rows})!=1 or len({r['pairs'] for r in pair_rows})!=1:
                    raise RuntimeError('native visual pair noise or actual compute mismatch')
        (args.output/f'lane{index}_terminal.json').write_text(json.dumps(rows,indent=2))
        return rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,i) for i in range(2)]
        all_rows=[row for f in futures for row in f.result()]
    ok=len(all_rows)==8 and all(r['returncode']==0 for r in all_rows)
    (args.output/'batch_terminal.json').write_text(json.dumps(dict(status='pass' if ok else 'fail',rows=all_rows),indent=2))
    sys.exit(0 if ok else 1)


if __name__=='__main__':main()
