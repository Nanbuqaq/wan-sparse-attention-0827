#!/usr/bin/env python3
"""Close the final native restoration/infrastructure chain without hiding failures."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();base=args.results/'metrics/sprint24h_20260907';rows=[];errors=[]
    names=['native_restore_benchmark_v1','native_restore_benchmark_v2','native_restore_benchmark_v3_recipe',
           'native_restore_benchmark_v4_recorded_recipe','native_restore_benchmark_v5_hybrid']
    for name in names:
        path=base/name/'summary.json'
        try:
            data=json.loads(path.read_text());status=data['status']
            if status not in ('pass','fail','negative'):raise ValueError('nonterminal status')
            rows.append(dict(name=name,status=status,source=str(path.resolve()),sha256=sha(path),
                offline_witness_recipe_search=data.get('recipe_selected_with_offline_witness'),
                medians_accepted=data.get('full_original_KV_hash_gates',False) and status=='pass',
                full_original_KV_hashes=data.get('full_original_KV_hash_gates',False)))
        except (OSError,ValueError,KeyError) as error:errors.append(f'{name}: {error}')
    receipt=args.results/'infrastructure/inferhub/sprint24h_native_episode_5kpro_4263845/prep_failed_receipt'
    task='zhouhe08__longlive2_episode_5kpro_replication_Iter0__4263845f2ccb'
    infrastructure={}
    try:
        data=json.loads((receipt/(task+'.json')).read_text());log=(receipt/(task+'.log')).read_text()
        if data['phase']!='prep' or 'AttrsDescriptor' not in log:raise ValueError('unexpected platform failure')
        infrastructure=dict(task=task,status='prep_failed',GPU_generations_started=0,
            gate_generations_not_started=8,conditional_full_generations_not_started=16,
            public_environment_not_modified=True,receipt_sha256=sha(receipt/(task+'.json')),
            log_sha256=sha(receipt/(task+'.log')),not_a_method_quality_failure=True)
    except (OSError,KeyError,ValueError) as error:errors.append(f'Blackwell receipt: {error}')
    refusals=[]
    root=args.results/'videos/sprint24h_20260907/longlive2_native_placement_gate64_v1'
    for lane,arm in ((0,'global'),(0,'shot'),(1,'global'),(1,'none')):
        path=root/f'lane{lane}'/(arm+'.log')
        try:
            if 'no idle unlocked local GPU' not in path.read_text():raise ValueError('unexpected initial launch log')
            if not (root/f'lane{lane}'/(arm+'.attempt2.log')).is_file():raise ValueError('missing recovery attempt log')
            refusals.append(dict(lane=lane,arm=arm,initial_status='prelaunch_resource_check_refused',
                original_log_sha256=sha(path),initial_GPU_generations=0,
                successful_prior_cases_not_rerun=True,recovery_source_frozen=True))
        except (OSError,ValueError) as error:errors.append(f'placement launch {lane}/{arm}: {error}')
    reviews=[]
    for name in ('longlive2_native_episode509_review_v1','longlive2_native_placement_gate64_review_v1'):
        path=base/name/'INTERPRETATION.md'
        if path.is_file():reviews.append(dict(name=name,source=str(path.resolve()),sha256=sha(path),assistant_qualitative_not_blind_human=True))
        else:errors.append(f'missing completed review: {name}')
    result=dict(status='pass' if not errors else 'fail',scope='final_native_boundary_chain_completeness_not_method_success',
                restoration_attempts=rows,optional_hardware=infrastructure,placement_prelaunch_refusals=refusals,
                semantic_reviews=reviews,errors=errors,video_inventory_is_separate=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
