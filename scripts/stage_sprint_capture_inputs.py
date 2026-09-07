#!/usr/bin/env python3
"""Copy eight existing complete captures to a new portable, SHA-locked input bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--verify-manifest', type=Path)
    args = p.parse_args()
    if args.verify_manifest:
        manifest = json.loads(args.verify_manifest.read_text())
        root = args.verify_manifest.parent.resolve()
        for row in manifest['captures']:
            path = (root/row['file']).resolve()
            if not path.is_relative_to(root) or sha(path) != row['sha256']:
                raise ValueError('portable input integrity failed')
        if {(r['kind'], r['layer']) for r in manifest['captures']} != {
                (k, l) for k in ('motion', 'state') for l in (0, 9, 19, 29)}:
            raise ValueError('portable input cohort incomplete')
        print(json.dumps({'status': 'verified', 'manifest_sha256': sha(args.verify_manifest)}))
        return
    if args.workspace is None or args.output is None:
        p.error('workspace/output or verify-manifest required')
    args.output.mkdir(parents=True, exist_ok=False)
    audit_path = args.workspace/'results/metrics/matched_trajectory_capture_be00491/trajectory_audit.json'
    audit = json.loads(audit_path.read_text())
    rows = []
    for case in audit['cases']:
        if case['method'] != 'rag_dense':
            continue
        kind = case['prompt'].removeprefix('calibration_')
        if kind not in {'motion', 'state'}:
            raise ValueError('only the two declared development categories may be staged')
        for layer in (0, 9, 19, 29):
            source = Path(case['capture_dir'])/f'layer{layer:02d}_start00046800_pass00.pt'
            if not source.resolve().is_relative_to((args.workspace/'results').resolve()):
                raise ValueError('capture source outside the declared research results')
            target = args.output/f'{kind}_layer{layer:02d}.pt'
            before = sha(source)
            shutil.copyfile(source, target)
            if sha(target) != before:
                raise RuntimeError('capture copy SHA mismatch')
            rows.append(dict(kind=kind, layer=layer, file=target.name, sha256=before,
                             source=str(source), bytes=target.stat().st_size))
    if len(rows) != 8:
        raise ValueError('expected exactly two categories and four layers')
    result = dict(status='frozen_existing_complete_capture_inputs', captures=rows,
                  parent_audit_sha256=sha(audit_path), new_generation=False)
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status':'pass', 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'manifest_sha256': sha(args.output/'manifest.json')}))


if __name__ == '__main__':
    main()
