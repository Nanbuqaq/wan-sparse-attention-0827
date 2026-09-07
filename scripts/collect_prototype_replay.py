#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--exit-codes', nargs=4, type=int, required=True)
    args = p.parse_args()
    records = []
    for lane, code in enumerate(args.exit_codes):
        path = args.root/f'lane{lane}/results/summary.json'
        payload = json.loads(path.read_text()) if path.exists() else dict(status='fail', reason='lane failed before summary', cases=[])
        valid = code == 0 and payload['status'] == 'pass' and len(payload.get('cases', [])) == 2
        row = dict(lane=lane, exit_code=code, status='pass' if valid else 'fail',
                   completed_captures=len(payload.get('cases', [])), result=payload)
        (args.root/f'lane{lane}/terminal.json').write_text(json.dumps(row, indent=2)+'\n')
        records.append(row)
    result = dict(status='pass' if all(r['status'] == 'pass' for r in records) else 'fail',
                  missing_lane_terminals=0, lanes=records, scope='offline_GPU_teacher_replay_not_video_or_speed')
    (args.root/'batch_terminal.json').write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
