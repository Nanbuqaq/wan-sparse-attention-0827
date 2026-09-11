#!/usr/bin/env python3
"""Real GPU equivalence and state-preservation gate for reusable source extraction."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_raw_source_input import load_raw_source_window
from adapters.longlive_sparse.cached_source_geometry import CachedSourceGeometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--expected-masks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    summary=json.loads((args.case/'summary.json').read_text())
    windows={i:load_raw_source_window(args.case/'source_raw_rgb.pt',i,summary) for i in (1,2,3)}
    expected=np.load(args.expected_masks)['pixel_masks'];cpu_rng=torch.random.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    model=CachedSourceGeometry(args.checkpoint);rows=[];last=None
    for attempt,version in enumerate((2,1,3,2)):
        try:
            result=model.extract(windows[version]);row={k:v for k,v in result.items() if k not in ('indices','token_masks','pixel_masks','records')}
            row['source_tokens']=result['indices'].numel()
            if version==2:
                row['exact_pixel_masks_to_existing_reference']=np.array_equal(result['pixel_masks'].numpy(),expected)
                if not row['exact_pixel_masks_to_existing_reference']:
                    torch.save(result,args.output/f'mismatch_{attempt}.pt')
                    (args.output/'failure.json').write_text(json.dumps(row,indent=2)+'\n')
                    raise RuntimeError('shared model changed source masks')
                if last is not None and not torch.equal(last,result['indices']):raise RuntimeError('intervening source changed cached result')
                last=result['indices'].clone()
            rows.append(row)
        except ValueError as e:
            if version==2:raise
            rows.append(dict(archive_version=version,status='no_registered_geometry',error=str(e)))
    rng_exact=torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    if not rng_exact:raise RuntimeError('auxiliary geometry altered RNG state')
    report=dict(status='pass',rows=rows,model_load_s=model.load_s,checkpoint_verify_CPU_s=model.checkpoint_verify_s,
        model_tensor_bytes=model.model_tensor_bytes,
        checkpoint_sha256=model.checkpoint_sha,CPU_and_GPU_RNG_unchanged=True,GPU=torch.cuda.get_device_name(),
        queue_or_live_routing_not_yet_implemented=True)
    (args.output/'gate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
