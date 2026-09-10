#!/usr/bin/env python3
"""Recover an existing twenty-case study; never submit or rewrite remote state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha(path):
    with path.open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    terminal = args.source / 'memory_study_terminal.json'
    expected = '811af4075a233c0afd841dcb4ee51cd83fc915cbe739a22606ca2b5d985f4848'
    if sha(terminal) != expected:
        raise RuntimeError('source terminal changed; inspect before recovery')
    shutil.copy2(terminal, args.output / terminal.name)
    rows = []
    started = time.monotonic()
    for summary in sorted(args.source.glob('lane*/*/summary.json')):
        rel = summary.parent.relative_to(args.source)
        dest = args.output / rel
        dest.mkdir(parents=True, exist_ok=False)
        row = dict(case=str(rel), files=[], status='running')
        try:
            data = json.loads(summary.read_text())
            assert data['status'] == 'pass'
            assert data['runner_commit'] == 'de72255cbf26b696f7f1d3c22c8c2f38db33266e'
            for name in ('summary.json', 'latents.pt', 'video.mp4', 'config.yaml'):
                source = summary.parent / name
                if name == 'config.yaml' and not source.exists():
                    continue
                before = sha(source)
                shutil.copy2(source, dest / name)
                after = sha(dest / name)
                if before != after:
                    raise RuntimeError('payload copy mismatch: ' + name)
                row['files'].append(dict(name=name, bytes=source.stat().st_size, sha256=after))
            row['status'] = 'pass'
        except Exception as error:
            row.update(status='fail', error=repr(error))
        rows.append(row)
        (args.output / 'recovery.json').write_text(json.dumps(dict(
            source=str(args.source), terminal_sha256=expected, rows=rows,
            complete=False, semantic_review_complete=False), indent=2) + '\n')
        print(json.dumps(dict(case=str(rel), status=row['status'])), flush=True)
    report = dict(source=str(args.source), terminal_sha256=expected, rows=rows,
                  complete=len(rows) == 20 and all(r['status'] == 'pass' for r in rows),
                  semantic_review_complete=False, elapsed_s=time.monotonic()-started)
    (args.output / 'recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
