#!/usr/bin/env python3
"""Retrospective execution inventory, not a semantic success or selection freeze.

Counts include repeated equivalent trajectories and branch gates. The prior
formal42 and Tether cohorts remain in their own audits; they are not regenerated
or silently pooled here. This inventory is scoped to the named sprint cohorts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


EXPECTED={
    'committed_moments153_v1':10,'committed_moments477_seed08_v1':8,
    'direct_rope_stream_trace_v1':1,'episode_anchor477_v1':6,'episode_anchor_gate39_v1':3,
    'episode_snapshot39_gate_v1':3,'episode_snapshot477_h200_v1':6,
    'event_retrieval477_h800_v1':10,'event_retrieval_gate39_v1':3,
    'group153_v1':10,'group477_h_v1':10,'group_budget477_v1':14,
    'local_only_revisit477_v1':4,'local_window8_precision153_v1':8,
    'metadata153_v1':8,'metadata477_h800_v1':8,'official_interactive477_h_v1':8,
    'official_interactive_local_gate39_v1':4,'precision_wire153_v1':10,
    'prototype_reference153_v1':8,'recache_version_capture39_v1':4,
    'revisit_dense477_h_v1':4,'revisit_runtime_gate_v1':1,'revisit_runtime_gate_v2':3,
    'rope_streaming153_gate_v1':4,'streaming_factorial477_h200_v1':48,
    'streaming_pipeline153_v1':8,'streaming_priority153_v2':6,'streaming957_local_v1':12,
    'longlive2_native_gate24_v1':1,'past_text39_gate_v1':2,'past_text477_local_v1':4,
    'longlive2_native509_h_v1':4,
    'longlive2_native_cut_gate48_v1':1,'longlive2_native_cut509_h_v1':4,
}


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def inspect_cohort(root,expected,*,allow_active=False,hash_payload=False):
    paths=sorted(list(root.rglob('case_state.json'))+list(root.rglob('terminal.json')))
    if root.name.startswith('longlive2_native'):
        # This independent native runner has one execution per summary, not a
        # variants aggregate. Its failure summary is also terminal evidence.
        paths+=sorted(root.rglob('summary.json'))
    rows=[];errors=[];seen=set()
    for path in paths:
        state=json.loads(path.read_text())
        if 'states' in state:continue  # Aggregate is not another execution.
        if path.parent in seen:raise ValueError('duplicate terminal files for a case')
        seen.add(path.parent);status=state.get('status')
        if status not in ('pass','negative','fail'):errors.append(str(path)+': nonterminal status')
        video=path.parent/'video.mp4';latent=path.parent/'latents.pt'
        if status in ('pass','negative') and (not video.is_file() or not latent.is_file()):
            errors.append(str(path)+': successful execution lacks video/latents')
        artifacts={}
        for label,file in (('terminal',path),('video',video),('latents',latent)):
            if file.is_file():
                    artifacts[label]=dict(path=str(file.resolve()),bytes=file.stat().st_size,
                    sha256=sha(file) if label=='terminal' or hash_payload else None)
        rows.append(dict(case=str(path.parent.relative_to(root)),terminal_status=status,
            artifacts=artifacts,semantic_success_not_inferred=True,
            declared_source_commit=state.get('execution_commit',state.get('commit',state.get('source_commit',state.get('runner_commit'))))))
    videos=list(root.rglob('*.mp4'))
    unowned=[str(p.relative_to(root)) for p in videos if p.parent not in seen]
    remaining=expected-len(rows)
    if remaining<0:errors.append('more executions than expected inventory')
    if not allow_active and remaining:errors.append(f'{remaining} executions missing terminal states')
    if not allow_active and unowned:errors.append('video without terminal ownership')
    return dict(cohort=root.name,expected_launched_executions=expected,terminal_executions=len(rows),
        remaining=remaining,counts=dict(Counter(r['terminal_status'] for r in rows)),
        video_files=len(videos),unowned_video_files=unowned,errors=errors,rows=rows,
        status='fail' if errors else ('running' if remaining else 'pass'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--active-cohort',action='append',default=[]);p.add_argument('--hash-payloads',action='store_true')
    args=p.parse_args()
    unknown=set(args.active_cohort)-set(EXPECTED)
    if unknown:raise ValueError('unregistered active cohort: '+str(unknown))
    groups=[inspect_cohort(args.root/name,count,allow_active=name in args.active_cohort,
        hash_payload=args.hash_payloads and name not in args.active_cohort) for name,count in EXPECTED.items()]
    extras=[p.name for p in args.root.iterdir() if p.is_dir() and p.name not in EXPECTED]
    report=dict(status='fail' if any(g['errors'] for g in groups) or extras else
        ('in_progress' if any(g['remaining'] for g in groups) else 'pass'),
        scope='named_sprint_video_execution_inventory_not_whole_research_completion',
        retrospective_inventory_not_preregistered_selection=True,repeated_trajectories_and_gates_included=True,
        prior_formal_and_Tether_results_not_pooled=True,semantic_quality_requires_separate_reviews=True,
        expected_executions=sum(EXPECTED.values()),terminal_executions=sum(g['terminal_executions'] for g in groups),
        remaining=sum(g['remaining'] for g in groups),unregistered_cohorts=extras,groups=groups,
        payload_hashes_requested=args.hash_payloads,active_cohorts_not_payload_hashed=args.active_cohort)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'}))
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':main()
