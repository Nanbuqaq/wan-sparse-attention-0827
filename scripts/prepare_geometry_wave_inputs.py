#!/usr/bin/env python3
"""Prepare pinned SAM2 dependencies in this job's writable output, using no GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

SOURCE_COMMIT='2b90b9f5ceec907a1c18123530e92e794ad901a4'
SOURCE_SHA='1f2fbfad3ffa38110368abac76c6ef9df9c282a66d5c2807bc94abf4d2fb30f8'
WEIGHT_SHA='7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b'


def reference_payload(hashes):
    values=hashes.split(':')
    if len(values)!=3 or any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in values):
        raise ValueError('three exact reference SHA256 values required')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from scripts.run_longlive2_native_reference import native_cut_schedule
    _,prompts=native_cut_schedule(root,'generated_patchwork_toy_cut_revisit')
    return dict(status='pass',seed=20260913,latent_shape=[1,128,48,44,80],prompts_per_block=prompts[0],
        noise_sha256=values[0],latent_sha256=values[1],pixels=dict(raw_RGB_sha256=values[2]))


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def fetch(url,path,expected):
    if not path.exists():
        partial=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(url,timeout=120) as source,partial.open('xb') as target:
            shutil.copyfileobj(source,target,8*1024**2)
        if digest(partial)!=expected:raise ValueError('download hash differs: '+path.name)
        partial.rename(path)
    if digest(path)!=expected:raise ValueError('existing download hash differs: '+path.name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    # Only hashes are passed via private job metadata; no research report enters Git.
    for method,variable in (('geometry_all32','GEOMETRY_ALL32_REF'),('geometry_holdfirst','GEOMETRY_HOLDFIRST_REF')):
        (out/(method+'_reference.json')).write_text(json.dumps(reference_payload(os.environ[variable]),indent=2)+'\n')
    archive=out/'sam2_source.tar.gz'
    fetch('https://codeload.github.com/facebookresearch/sam2/tar.gz/'+SOURCE_COMMIT,archive,SOURCE_SHA)
    with tarfile.open(archive) as package:
        package.extractall(out/'source',filter='data')
    shutil.copytree(out/'source'/('sam2-'+SOURCE_COMMIT)/'sam2',out/'sam2-runtime/sam2')
    fetch('https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',out/'sam2_hiera_large.pt',WEIGHT_SHA)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',
        '--target',str(out/'sam2-python'),'hydra-core==1.3.2','iopath==0.1.10','portalocker==3.2.0'],check=True)
    # Image-only prediction with zero hole/sprinkle cleanup does not invoke SAM's CUDA extension.
    rows=[dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=digest(path))
        for path in sorted(out.rglob('*')) if path.is_file() and '__pycache__' not in path.parts]
    (out/'manifest.json').write_text(json.dumps(dict(schema='geometry_wave_private_inputs_v1',files=rows,
        source_commit=SOURCE_COMMIT,CUDA_extension_unused_for_zero_cleanup_image_path=True),indent=2)+'\n')
    print(json.dumps(dict(status='pass',files=len(rows),GPU_work_not_started=True)))


if __name__=='__main__':main()
