#!/usr/bin/env python3
"""Frozen local/InferHub lanes; successful cases are never regenerated."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'third_party/LongLive2')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('gate','screen'),required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--component-gate',action='store_true')
    p.add_argument('--required-gpu-name',default='')
    args=p.parse_args()
    spec_path=ROOT/'configs/system/native_resident_wave1.json';spec=json.loads(spec_path.read_text())
    sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    visible=[s for s in visible if s]
    if args.run and not visible:raise RuntimeError('require assigned GPUs or local physical card locks')
    cases=[]
    arms=('native','identity') if args.stage=='gate' else spec['methods']
    scenarios=spec['scenarios'][1:] if args.stage=='gate' else spec['scenarios']
    for scenario in scenarios:
        for arm in arms:
            name=scenario+'__'+arm
            cmd=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),
                '--assets',str(args.assets),'--source',str(args.source),'--output',str(args.output/name),
                '--cut-scenario',scenario,'--seed',str(spec[args.stage+'_seed']),
                '--native-local-frames','32','--cfg1-positive-cache-only',
                '--fixed-adaln-warps','16','--fixed-adaln-stages','1',
                '--constructor-mode','strict_checkpoint_no_parameter_init']
            if args.stage=='gate':cmd+=['--gate','--episode-gate-layout']
            if arm!='native':
                cmd+=['--resident-history-policy',arm.replace('_reuse',''),
                      '--resident-history-fraction',str(spec['fraction'])]
                if arm.endswith('_reuse'):cmd+=['--resident-history-reuse','denoise_first']
            cases.append(dict(id=name,arm=arm,scenario=scenario,cmd=cmd))
    plan=dict(code_sha=sha,config_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
              stage=args.stage,cases=cases,visible_devices=visible,source=str(args.source))
    if not args.run:
        print(json.dumps(plan,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'batch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def lane(index):
        rows=[]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=visible[index]
        for key in ('WAN_SPARSE_PHYSICAL_GPU','WAN_SPARSE_PHYSICAL_GPUS'):env.pop(key,None)
        env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',
            LLV2_USE_FA3='0',LLV2_USE_FA4='0',LLV2_USE_TE_ATTN='0',LLV2_COMPILE_VAE='0',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if args.component_gate:
            target=args.output/f'component_lane{index}'
            cmd=[sys.executable,str(ROOT/'scripts/gate_native_resident_component.py'),
                 '--output',str(target),'--required-gpu-name',args.required_gpu_name]
            with (args.output/f'component_lane{index}.log').open('x') as handle:
                gate_code=subprocess.call(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if gate_code:
                return [dict(id=c['id'],lane=index,returncode=gate_code,status='blocked_by_component_gate',
                             component_report=str(target/'gate.json')) for c in cases[index::len(visible)]]
        for case in cases[index::len(visible)]:
            log=args.output/(case['id']+'.log')
            summary=args.output/case['id']/'summary.json'
            try:
                with log.open('x') as handle:
                    code=subprocess.call(case['cmd'],env=env,stdout=handle,stderr=subprocess.STDOUT)
                data=json.loads(summary.read_text()) if summary.exists() else {}
                row=dict(id=case['id'],lane=index,returncode=code,status=data.get('status','missing'),
                         summary=str(summary),log=str(log))
            except Exception as error:
                row=dict(id=case['id'],lane=index,returncode=-1,status='fail',error=repr(error),log=str(log))
            rows.append(row)
            print(json.dumps(rows[-1]),flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=min(len(visible),len(cases))) as pool:
        rows=[row for result in pool.map(lane,range(min(len(visible),len(cases)))) for row in result]
    terminal=dict(code_sha=sha,stage=args.stage,rows=rows,
                  status='pass' if all(r['returncode']==0 and r['status']=='pass' for r in rows) else 'fail',
                  quality_review_complete=False)
    if args.stage=='gate' and terminal['status']=='pass':
        values=[json.loads((args.output/c['id']/'summary.json').read_text()) for c in cases]
        equal=all(values[0][k]==values[1][k] for k in ('noise_sha256','latent_sha256','native_shot_pin_events'))
        equal=equal and values[0]['pixels']['raw_RGB_sha256']==values[1]['pixels']['raw_RGB_sha256']
        terminal['identity_full_noise_latent_RGB_pin_exact']=equal
        import torch
        import av
        torch.set_num_threads(2)
        actual=[torch.load(args.output/c['id']/'latents.pt',map_location='cpu',weights_only=True) for c in cases]
        terminal['actual_complete_latents_equal']=torch.equal(*actual)
        decoded=[]
        for c in cases:
            digest=hashlib.sha256();count=0
            with av.open(str(args.output/c['id']/'video.mp4')) as container:
                container.streams.video[0].codec_context.thread_count=2
                for frame in container.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
            decoded.append(dict(frames=count,sha256=digest.hexdigest()))
        terminal['decoded_RGB_audit']=decoded
        equal=equal and terminal['actual_complete_latents_equal'] and decoded[0]==decoded[1] and decoded[0]['frames']==253
        if not equal:terminal['status']='fail'
    (args.output/'batch_terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    if terminal['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
