#!/usr/bin/env python3
"""A bounded execution inventory, not a count of independent scientific samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

EXPECTED={'gate64_v1':5,'capacity_gate64_v1':3,'same_compiler_gate64_v1':1,
          'context_gate64_v1':2,'context509_local_v1':8,'attention_teacher509_v1':2,
          'attention_teacher509_v2':2,'native_capacity_5kpro_v1':16,
          'initial_anchor_gate64_v1':2,'initial_anchor509_local_v1':4,'source_pin_lifetime509_v1':2,
          'semantic_remat_gate64_v1':2,'semantic_remat509_v1':2,'settled_state_screen509_v1':4,'continuation_screen509_v1':4}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    groups=[];rows=[]
    for cohort,expected in EXPECTED.items():
        found=[]
        for path in sorted((args.root/cohort).rglob('summary.json')):
            d=json.loads(path.read_text());entry=dict(case=str(path.parent.relative_to(args.root)),status=d['status'],
                GPU=d.get('gpu'),runner_commit=d.get('runner_commit'),seed=d.get('seed'),latent_shape=d.get('latent_shape'),
                has_video=(path.parent/'video.mp4').is_file(),has_latents=(path.parent/'latents.pt').is_file(),
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            found.append(entry)
        groups.append(dict(cohort=cohort,expected=expected,found=len(found),missing=max(0,expected-len(found)),cases=found));rows+=found
    counts=Counter(r['status'] for r in rows)
    result=dict(expected_executions=sum(EXPECTED.values()),found_executions=len(rows),statuses=dict(counts),
        missing=sum(g['missing'] for g in groups),groups=groups,
        independent_scientific_samples_not_equal_to_executions=True,
        excluded_unstarted_H20=dict(job_id='zhouhe08__longlive2_memory_placement_lifetime_capacity_Iter0__de72255cbf26',
            state='pending_after_platform_requeue_check_live_for_changes',planned_cases=20),
        separate_cut_component_phase=dict(expected=6,terminal_reports_found=len(list((args.root/'cut_components509_v1').glob('lane*/*/summary.json')))),
        overall_research_goal_complete=False)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))


if __name__=='__main__':main()
