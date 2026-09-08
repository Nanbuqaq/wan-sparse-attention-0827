#!/usr/bin/env python3
"""Copy only verified native reference assets and private Python overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.fetch_longlive2_reference_assets import FILES,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--overlay',type=Path,required=True);args=p.parse_args()
    manifest=args.source/'assets_manifest.json';loaded=json.loads(manifest.read_text())
    if loaded['status']!='pass':raise ValueError('verified source assets required')
    args.destination.mkdir(parents=True,exist_ok=False)
    records=[]
    by_key={(r['repo'],r['file']):r for r in loaded['files']}
    for repo,revision,name,relative,_ in FILES:
        source=args.source/relative;target=args.destination/relative
        if sha256(source)!=by_key[repo,name]['sha256']:raise ValueError('source payload changed')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target,follow_symlinks=True)
        actual=sha256(target)
        if actual!=by_key[repo,name]['sha256']:raise ValueError('staging copy changed')
        records.append(dict(relative=relative,sha256=actual,bytes=target.stat().st_size))
    shutil.copytree(args.overlay,args.destination/'python-overlay',symlinks=False)
    shutil.copyfile(manifest,args.destination/'assets_manifest.json')
    report=dict(status='pass',assets=records,original_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        code_not_staged='all executable project and native model code must come from pinned Git commits',
        private_overlay_copied=True)
    (args.destination/'staging_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',assets=len(records),destination=str(args.destination))))


if __name__=='__main__':main()
