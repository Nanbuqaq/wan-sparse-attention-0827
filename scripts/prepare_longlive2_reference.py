#!/usr/bin/env python3
"""CPU-only integrity/import gate before native reference GPU assignment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    args=p.parse_args();manifest=json.loads((args.inputs/'staging_manifest.json').read_text())
    if manifest['status']!='pass':raise ValueError('incomplete staging')
    for row in manifest['assets']:
        path=(args.inputs/row['relative']).resolve()
        if not path.is_relative_to(args.inputs.resolve()):raise ValueError('asset escapes declared inputs')
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('staged asset corrupted')
    expected='6b36d20ec6f7958d29d11a704dfa64611a9f2572'
    if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=expected:
        raise ValueError('native source differs from GPU gate')
    sys.path.insert(0,str(args.source))
    import torch
    from pipeline import CausalDiffusionInferencePipeline
    from wan_5b.modules.attention import FLASH_ATTN_2_AVAILABLE
    if not FLASH_ATTN_2_AVAILABLE:raise RuntimeError('native FA2 is mandatory')
    print(json.dumps(dict(status='pass',source=expected,torch=torch.__version__,native_FA2=True,assets=len(manifest['assets']))))


if __name__=='__main__':main()
