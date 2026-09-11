#!/usr/bin/env python3
"""CPU-only private input integrity check before requesting accelerator lanes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    root=args.inputs.resolve()
    for row in manifest['files']:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('input manifest path leaves bundle')
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=row['sha256']:raise ValueError('private input checksum differs: '+row['path'])
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print(json.dumps(dict(status='pass',files=len(manifest['files']),GPU_work_not_started=True)))


if __name__=='__main__':main()
