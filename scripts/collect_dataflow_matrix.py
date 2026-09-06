#!/usr/bin/env python3
"""All frozen GPU lanes reach an explicit terminal state; profiler is separate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--exit-codes', nargs=8, type=int, required=True)
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for index in range(72):
        lane = index%8
        path = root/f'lane{lane}/case{index:02d}.json'
        if not path.is_file():
            if args.exit_codes[lane] == 0:
                raise ValueError('successful lane omitted a frozen case')
            path = root/f'case{index:02d}_failed_before_result.json'
            path.write_text(json.dumps({'status': 'fail', 'case': {'index': index},
                'failure_reason': f'GPU lane{lane} exited{args.exit_codes[lane]} before case result',
                'failure_scope': 'infrastructure_incomplete_not_method_quality'}, indent=2)+'\n')
        data = json.loads(path.read_text())
        if data['case']['index'] != index or data['status'] not in ('pass', 'fail', 'negative'):
            raise ValueError('case identity/terminal status mismatch')
        if data['status'] == 'pass' and (data['warmup'] != 5 or data['repeats'] != 30 or len(data['records']) != 10):
            raise ValueError('a passing matrix case did not execute the full frozen measurement protocol')
        rows.append({'index': index, 'lane': lane, 'status': data['status'], 'path': str(path),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'status': 'pass' if all(r['status'] == 'pass' for r in rows) else 'fail',
        'expected': 72, 'pass': sum(r['status'] == 'pass' for r in rows),
        'fail': sum(r['status'] == 'fail' for r in rows), 'missing': 0, 'records': rows,
        'lane_exit_codes': args.exit_codes, 'profiles_not_in_timing_matrix': True}
    (root/'terminal.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
