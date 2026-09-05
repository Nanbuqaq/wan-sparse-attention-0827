#!/usr/bin/env python3
"""Recover individual terminal cases even when a batch is killed before merge."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def recover(root, expected, platform, external):
    wanted={case['case_key_sha256']:case for case in expected['cases']}
    cases={}
    sources=[]
    for path in sorted(root.rglob('case_state.json'))+list(external):
        raw=json.loads(path.read_text())
        key=raw['case_key_sha256']
        if key not in wanted or raw['case_key']!=wanted[key]['case_key']:
            raise ValueError(f'unexpected or inconsistent identity: {path}')
        if key in cases:
            raise ValueError(f'duplicate case: {path}')
        source_sha=sha(path)
        state={**wanted[key],**raw,'recovered_source_state_sha256':source_sha,
               'recovery_change_scope':'artifact_path_relocation_only'}
        for field,name in (('video','video.mp4'),('stats','sparse_history_stats.json'),('config','case_config.json')):
            if (path.parent/name).is_file():
                state[field]=str((path.parent/name).resolve())
        cases[key]=state
        sources.append({'path':str(path.resolve()),'sha256':source_sha})
    reason=platform.get('result',platform).get('reason')
    if len(cases)!=len(wanted) and not reason:
        raise ValueError('cannot invent terminal failures without a terminal platform cause')
    missing=[]
    for key,wanted_case in wanted.items():
        if key not in cases:
            missing.append(key)
            cases[key]={**wanted_case,'status':'fail',
                'failure_reason':f'batch terminated with {reason} before case terminal state',
                'failure_scope':'runtime_incomplete_not_algorithm_quality',
                'platform_job_id':platform['job_id']}
    return {'cases':list(cases.values()),'sources':sources,'recovered_case_artifacts':len(sources),
            'synthesized_failure_states':missing,'platform_reason':reason,
            'missing_terminal_states':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--expected',required=True)
    parser.add_argument('--platform',required=True)
    parser.add_argument('--external-case',action='append',default=[])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=recover(Path(args.root),json.loads(Path(args.expected).read_text()),
                   json.loads(Path(args.platform).read_text()),[Path(path) for path in args.external_case])
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(result['cases']),'recovered_artifacts':result['recovered_case_artifacts'],
                      'terminal_failures':len(result['synthesized_failure_states']),'missing':0}))


if __name__=='__main__':
    main()
