#!/usr/bin/env python3
"""One assigned GPU, independent frozen tasks and explicit per-task terminals."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.replay_cross_trajectory import evaluate
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def run_task(root,task,output):
    path=root/'plans'/task['plan_file']
    if sha256(path)!=task['plan_file_sha256']:raise ValueError('frozen plan SHA mismatch')
    plans={k:HistoryRoutePlan.from_state_dict(v) for k,v in torch.load(path,map_location='cpu',weights_only=True)['plans'].items()}
    captures=[]
    for name,sha in zip(task['capture_files'],task['capture_sha256']):
        p=root/name
        if sha256(p)!=sha:raise ValueError('frozen capture SHA mismatch')
        captures.append(torch.load(p,map_location='cpu',weights_only=True))
    start=time.perf_counter()
    result=evaluate(captures,plans,actor=task['actor'])
    result.update(task_id=task['task_id'],prompt=task['prompt'],gpu=torch.cuda.get_device_name(),
        cuda_capability=list(torch.cuda.get_device_capability()),torch_version=torch.__version__,
        scope='fixed_input_fp32_replay_not_video_generation_or_speed_trial',diagnostic_wall_s=time.perf_counter()-start)
    with output.open('x') as f:json.dump(result,f,indent=2)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',required=True)
    parser.add_argument('--lane',required=True,type=int)
    parser.add_argument('--lanes',required=True,type=int)
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int,default=0,help='local real-GPU preflight only')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('real assigned GPU required')
    root=Path(args.input_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ready=json.loads((root/'READY.json').read_text())
    if sha256(root/'manifest.json')!=ready['manifest_sha256']:raise ValueError('manifest SHA mismatch')
    tasks=json.loads((root/'manifest.json').read_text())['tasks'][args.lane::args.lanes]
    if args.limit:tasks=tasks[:args.limit]
    states=[]
    for task in tasks:
        output=out/f'{task["task_id"]}.json'
        try:
            result=run_task(root,task,output)
            row={'task_id':task['task_id'],'status':'pass','output':str(output)}
        except Exception as e:
            row={'task_id':task['task_id'],'status':'fail','error':str(e)}
            (out/f'{task["task_id"]}.failure.json').write_text(json.dumps({**row,'traceback':traceback.format_exc()},indent=2)+'\n')
        states.append(row)
        (out/'states.json').write_text(json.dumps({'expected':len(tasks),'cases':states},indent=2)+'\n')
        print(json.dumps(row),flush=True)
    terminal={'status':'pass' if all(r['status']=='pass' for r in states) else 'fail',
              'expected':len(tasks),'terminal':len(states),'missing':len(tasks)-len(states),
              'cases':states,'gpu':torch.cuda.get_device_name()}
    (out/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    if terminal['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
