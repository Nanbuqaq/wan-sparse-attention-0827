#!/usr/bin/env python3
"""Record actual assigned CUDA topology before any model case starts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_hardware import validate_hardware_names


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--expected-count',type=int,required=True)
    p.add_argument('--required',default='H200');p.add_argument('--allow-h800',action='store_true')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    result=dict(names=names,torch=torch.__version__)
    try:
        result.update(validate_hardware_names(names,required=args.required,allow_h800=args.allow_h800,expected_count=args.expected_count),status='pass')
    except ValueError as error:result.update(status='fail',error=str(error))
    with args.output.open('x') as h:json.dump(result,h,indent=2);h.write('\n')
    print(json.dumps(result),flush=True)
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
