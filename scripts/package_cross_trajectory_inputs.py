#!/usr/bin/env python3
"""CPU-only immutable data packaging; no source code or GPU allocation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe bundle-relative path')
    return p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-root',required=True)
    parser.add_argument('--plans-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepare-commit',required=True)
    parser.add_argument('--local-links',action='store_true',help='local-only read links; never an InferHub upload')
    args=parser.parse_args()
    root=Path(args.capture_root).resolve();plans=Path(args.plans_root).resolve();out=Path(args.output).resolve()
    if json.loads((root/'trajectory_audit.json').read_text())['status']!='pass':raise ValueError('trajectory gate failed')
    out.mkdir(parents=True,exist_ok=False)
    tasks=[];copied={}
    for kind in ('motion','state'):
        data=json.loads((plans/kind/'tasks.json').read_text())
        if data['status']!='pass':raise ValueError('plan preparation failed')
        for task in data['tasks']:
            pp=plans/kind/task['plan_file']
            target=out/'plans'/task['plan_file']
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(pp,target)
            if sha256(target)!=task['plan_file_sha256']:raise ValueError('copied plan SHA mismatch')
            for name,expected in zip(task['capture_files'],task['capture_sha256']):
                if name in copied:
                    if copied[name]!=expected:raise ValueError('conflicting capture hashes')
                    continue
                relative=safe_relative(name)
                target=out/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                if args.local_links:
                    target.symlink_to((root/relative).resolve())
                else:
                    shutil.copy2(root/relative,target)
                if sha256(target)!=expected:raise ValueError('copied capture SHA mismatch')
                copied[name]=expected
            tasks.append(task)
            print(json.dumps({'packed':task['task_id'],'captures':len(copied)}),flush=True)
    tasks.sort(key=lambda r:r['task_id'])
    if len(tasks)!=48 or len(copied)!=240:raise ValueError('expected48 sequences/240 captures')
    manifest={'status':'pass','scope':'frozen_39_latent_cross_trajectory_data_only',
              'prepare_commit':args.prepare_commit,'tasks':tasks,'capture_files':len(copied),
              'local_links_not_shared_worker_inputs':args.local_links}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(root/'trajectory_audit.json',out/'trajectory_audit.json')
    (out/'READY.json').write_text(json.dumps({'status':'pass','manifest_sha256':sha256(out/'manifest.json'),
                                            'task_count':48,'captures':240},indent=2)+'\n')


if __name__=='__main__':main()
