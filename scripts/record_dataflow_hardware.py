#!/usr/bin/env python3
"""Record real GPU/compiler identity before any kernel compilation can abort."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import torch
import triton


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--require-name', default='')
    args = p.parse_args()
    devices = []
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        devices.append({'logical_index': i, 'name': prop.name, 'total_memory': prop.total_memory,
                        'SM_count': prop.multi_processor_count, 'compute_capability': [prop.major, prop.minor]})
    passing = bool(devices) and all(args.require_name.lower() in d['name'].lower() for d in devices)
    payload = {'status': 'pass' if passing else 'fail', 'devices': devices, 'required_name': args.require_name,
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'triton': triton.__version__,
        'triton_module': triton.__file__, 'host': platform.node(),
        'assigned_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'failure_scope': None if passing else 'allocation_or_environment_not_method_quality'}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    print(json.dumps(payload), flush=True)
    if not passing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
