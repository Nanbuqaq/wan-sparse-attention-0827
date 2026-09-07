#!/usr/bin/env python3
"""Whole-video budget-control audit, retaining all costs and reused controls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_system_video_comparison import load_case,sha
from scripts.build_video_review_storyboards import analyze
from adapters.longlive_sparse.offline_eval import output_error_metrics


def variant(state):
    key=state['case_key']; method=key['method']
    if method=='rag_dense': return 'Dense'
    prefix='Final' if method=='transfer_vaware_hybrid_history' else key['method_params']['relation_admission']
    return prefix+str(round(100*key['history_density']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reused-root',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    expected=json.loads(args.expected.read_text())['cases']
    new_paths=list(args.root.rglob('case_state.json'))
    observed=[json.loads(path.read_text()) for path in new_paths]
    if {r['case_key_sha256'] for r in observed}!={r['case_key_sha256'] for r in expected} or len(observed)!=14:
        raise ValueError('frozen14-case set is incomplete or duplicated')
    grouped=defaultdict(dict)
    for path in [*args.reused_root.rglob('case_state.json'),*new_paths]:
        state=json.loads(path.read_text())
        if state['backend']!='resident_grouped_fa2': continue
        key=state['case_key']; tag=variant(state)
        group=(key['prompt_id'],key['seed'])
        if tag in grouped[group]: raise ValueError('duplicate variant in prompt-seed group')
        grouped[group][tag]=load_case(path.parent)
    rng=random.Random(613095)
    reports=[]; masked_manifest=[]
    for group,cases in sorted(grouped.items()):
        prompt,seed=group
        if not {'Dense','Final25','Final50','shared50','per_group25'}<=cases.keys():
            raise ValueError(f'incomplete high-budget control group: {group}')
        dense=cases['Dense']; names=sorted(cases); rng.shuffle(names)
        rows=[]
        for ordinal,name in enumerate(names):
            case,state,stats,config,latent,routes=cases[name]
            for field in ('prompt','seed','latent_frames','rope_policy','refresh_policy','backend'):
                if config[field]!=dense[3][field]: raise ValueError(f'incompatible control {field}')
            code=f'{prompt}_s{seed}__masked{ordinal:02d}'
            diagnostic=analyze(dict(id=code,video=str(case/'video.mp4'),latent_frames=config['latent_frames'],status='pass'),
                               args.output,samples_per_quarter=16)
            row=dict(variant=name,masked_code=code,case=str(case),case_identity=state['case_key_sha256'],
                source_commit=state['execution_commit'],state_sha256=sha(case/'case_state.json'),
                new_video=path_in(case,args.root),
                complete_wall_s=state['end_to_end_s'],inference_s=state['wall_breakdown']['inference_s'],
                history_pair_density=state['history_pair_density'],actual_history_H2D_bytes=state['transferred_bytes'],
                H2D_over_cumulative_candidate_bytes=state['history_transfer_density'],
                peak_GPU_allocated_GB=state['peak_allocated_gb'],
                fidelity_not_absolute_quality=dict(full=output_error_metrics(dense[4],latent),
                    late_quarter=output_error_metrics(dense[4][:,-30:],latent[:,-30:])),diagnostic=diagnostic,
                ordered_route_sha256=hashlib.sha256(json.dumps(routes).encode()).hexdigest())
            rows.append(row)
            masked_manifest.append(dict(id=code,overview=diagnostic['overview'],quarters=diagnostic['storyboards'],
                prompt=config['prompt'],rubric='identity/appearance,scene stability,task progress,temporal continuity,late-quarter retention; descriptive not automatic'))
        reports.append(dict(prompt=prompt,seed=seed,cases=rows,
            primary_same_hardware_comparison=all(row['new_video'] for row in rows),
            cross_hardware_diagnostic_only=not all(row['new_video'] for row in rows),
            hardware_boundary='new controls RTX4090; reused old-seed controls H-cluster; do not attribute cross-hardware timing or numerical drift to admission'))
    result=dict(status='pass',new_cases=14,missing=0,groups=reports,
        comparison='shared50 is a higher nominal raw-budget control; actual bytes measured, not assumed equal',
        timing_repeats=False,absolute_quality_winner=None,review_status='method_name_masked_AI_review_pending_not_independent_human')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'masked_review_manifest.json').write_text(json.dumps(masked_manifest,indent=2)+'\n')
    print(json.dumps(dict(status='pass',new_cases=14,groups=len(reports),masked_boards=len(masked_manifest))))


def path_in(case,root):
    return case.resolve().is_relative_to(root.resolve())


if __name__=='__main__':main()
