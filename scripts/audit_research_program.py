#!/usr/bin/env python3
"""Terminal evidence ledger; an expected environment failure is never a gain."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--test-log',type=Path,required=True)
    args=p.parse_args();work=args.workspace.resolve();repo=work/'publish_repo';metrics=work/'results/metrics'
    entries=[]
    def checked(name,path,predicate,outcome='pass',scope=''):
        data=json.loads(path.read_text())
        if not predicate(data):raise ValueError(f'program gate failed: {name}')
        entries.append({'id':name,'outcome':outcome,'path':str(path.resolve()),'sha256':sha(path),'scope':scope})
        return data
    holdout=repo/'configs/formal/system_holdout_prompts.json'
    freeze=repo/'configs/formal/system_method_freeze.json'
    if sha(holdout)!='6d898e96dd28e924622d5585ab6cc85b139560282c01761cde87b518f3d17fea':raise ValueError('formal prompt freeze changed')
    if sha(freeze)!='1ec149a7867f5761b1c66062aec5a4c4b29fb55d17f1dd074850d1db25ff2633':raise ValueError('formal method freeze changed')
    checked('holdout_freeze',holdout,lambda d:d['sparse_results_used'] is False and len(d['prompts'])==4)
    checked('method_freeze',freeze,lambda d:d['formal_results_used'] is False and len(d['configs'])==3)
    checked('dense_state_screen',repo/'docs/evidence/system_state_terminal_audit.json',lambda d:d['status']=='pass' and d['pass_cases']==4)
    checked('original_ten_case_calibration',work/'results/manifests/routing_calibration_4b6f976_full10_audit.json',
        lambda d:d['status']=='pass' and d['expected_cases']==d['terminal_cases']==10 and d['pass_cases']==9 and d['fail_cases']==1,
        scope='9 technical pass,1 preserved infrastructure failure; terminal completeness, not method promotion')
    checked('cost_model',metrics/'transfer_replay_83fb91a_h200_v2_summary.json',
        lambda d:d['status']=='pass' and d['cost_model_gate']['heldout_mape']>.15 and not d['cost_model_gate']['cost_aware_admission_allowed'],
        'negative','cost-aware runtime disabled by held-out prediction gate')
    checked('bootstrap_long_gate',metrics/'bootstrap477_quality/decision.json',
        lambda d:d['status']=='pass' and not d['formal_promotion'],'negative','long on-policy admission promotion rejected')
    checked('lossless_compiler_repeats',metrics/'shared_compiler_repeats_audit_20260906.json',
        lambda d:d['status']=='pass' and d['technical_successes']==8 and d['missing']==0)
    checked('hierarchical_video_gate',metrics/'hierarchical_pairs_93db4ab_audit.json',
        lambda d:d['status']=='pass' and d['formal_promotion'] is False,'negative','mixed full wall time, not promoted')
    checked('batched_backend_gate',metrics/'batched_backend_80ecdbe_audit.json',
        lambda d:d['status']=='pass' and d['equivalence_gate'] and not d['eligible_for_formal_system_config'],
        'negative','kernel-wrapper gain is not sufficient after a complete-time regression')
    checked('profile_scaling',metrics/'profile_scaling_report_v1/summary.json',
        lambda d:d['status']=='pass' and len(d['rows'])==6 and d['CPU_archive_bytes_per_additional_latent_120_to_240']==287539200.)
    checked('native_H200_grid',metrics/'dataflow_matrix_af26efc_h200/terminal.json',
        lambda d:d['status']=='pass' and d['pass']==72 and d['missing']==0)
    checked('native_H200_identity',metrics/'dataflow_matrix_af26efc_h200/hardware.json',
        lambda d:d['status']=='pass' and len(d['devices'])==8 and all('H200' in x['name'] for x in d['devices']))
    checked('three_hardware_references',metrics/'dataflow_three_hardware_v1/summary.json',
        lambda d:d['status']=='pass' and [x['cases'] for x in d['hardware']]==[72,12,12] and all(x['missing']==0 for x in d['hardware']))
    checked('H200_streaming_activity',metrics/'dataflow_matrix_af26efc_h200/profile/activity_audit.json',
        lambda d:d['status']=='pass' and d['transfers']['H2D']['overlap_with_GPU_kernels_s']>0,
        scope='actual CUDA activity overlap, not video or net-speed evidence')
    checked('Ncu_counter_attempt',metrics/'dataflow_counters_d31a3fa_h/terminal.json',
        lambda d:d['status']=='fail' and len(d['cases'])==3 and d['missing']==0 and all(r['counter_permission_denied'] for r in d['cases']),
        'fail','environment counter permission denied; HBM/L2/tile counters unavailable')
    checked('oracle_canonical_v2',metrics/'canonical_oracle_quality_v2/quality.json',
        lambda d:d['status']=='pass' and d['metric_input_protocol_version']==2 and len(d['rows'])==3,
        scope='offline full-candidate teacher only, excluded from online Pareto')
    checked('remaining_capture_noop_gates',metrics/'remaining_contracts_d4ce0bf/terminal.json',
        lambda d:d['status']=='pass' and len(d['cases'])==2 and all(r['same_latents_routes_noise_and_future_prototypes'] for r in d['cases']))
    checked('prefetch_budget_feedback_closure',metrics/'remaining_contracts_report_v2/summary.json',
        lambda d:d['status']=='pass' and d['full_capture_evaluations']==12 and d['ablation_comparisons']==72 and d['missing']==0
            and not d['new_route_promoted'] and not d['both_category_worst_case_nonregressing_variants'],
        'negative','completed development tests; direct prefetch and budget variants not promoted; some censored-reentry concerns observed')
    for label,count,groups in [('formal477',24,8),('formal957',12,4),('candle_stress',6,2)]:
        system_path=metrics/(f'{label}_58aa65b_group_audit.json' if label.startswith('formal') else 'candle_stress_8c1d77b_group_audit.json')
        checked(label+'_systems',system_path,lambda d,n=count,g=groups:d['status']=='pass' and d['technical_pass']==n and d['missing']==0
            and len(d['groups'])==g and all(r['same_final_latents'] and r['same_final_routes'] and r['same_final_preencode_pixels'] for r in d['groups']))
        visual_path=metrics/(f'{label}_visual_audit_v1.json' if label.startswith('formal') else 'candle_visual_audit_v1.json')
        checked(label+'_visual',visual_path,lambda d,n=count:d['status']=='pass' and d['reviewed_cases']==n and d['missing']==0,
                scope='complete unblinded assistant audit, not universal task success')
    videos=checked('final_video_integrity',metrics/'final_video_index_v1/index.json',
        lambda d:d['status']=='pass' and d['formal_videos']==36 and d['stress_videos']==6 and d['videos']==42 and d['missing']==0)
    for row in videos['rows']:
        for artifact in row['artifacts'].values():
            if sha(artifact['path'])!=artifact['sha256']:raise ValueError('final video/metric artifact changed')
    test_text=args.test_log.read_text()
    if 'failed' in test_text.split('short test summary info')[-1] or 'passed, 1 skipped' not in test_text:
        raise ValueError('final CPU regression not passing')
    baseline='59020c0610cb48ffdf0ce36c32aba97378aa0b72'
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',baseline,'HEAD'],check=True)
    source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    result={'status':'pass','scope':'terminal_research_execution_with_negative_and_environment_failure_outcomes',
        'formal_generation_cases':36,'stress_generation_cases':6,'formal_system_equivalent_pairs':12,
        'unique_reviewed_trajectories':28,'quarter_storyboards_reviewed':112,
        'missing_terminal_cases_in_final_cohorts':0,'all_listed_evidence_locks_valid':True,
        'evidence':entries,'baseline_commit':baseline,'audit_source_commit':source,
        'final_test_log':{'path':str(args.test_log.resolve()),'sha256':sha(args.test_log),'summary':test_text.strip().splitlines()[-1]},
        'selected_paper_direction':'lossless_system_execution_plus_characterization',
        'A_new_admission_plus_system_established':False,'B_causal_three_role_method_established':False,
        'C_adaptive_QOut_KVOut_video_established':False,'better_than_Dense_absolute_quality_established':False,
        'LongLive2_new_comparison_performed':False,'training_performed':False,
        'conditional_stops':{'cost_aware_runtime':'failed15%MAPE gate','causal_role_and_three_role_video':'role reliability/semantic evidence gates not passed',
            'optimized_KVOut_video':'Attention kernel below10% real profiled critical window; reference results did not justify promotion',
            'timed_q_to_next_prefetch':'near-random direct raw-space forecast, high extra traffic; stopped at screen, no timeliness claim',
            'adaptive_dataflow':'Q-stationary family wins>=90% in each tested scope'},
        'evidence_not_obtained':['privileged HBM/L2/on-chip hardware counters','causal semantic deletion proof for identity/scene/state',
            'bounded total CPU archive','deployed real-time infinite streaming service'],
        'invalidated_early_outputs':['canonical_quality_oracle_smoke_v1 quality metrics','canonical_frozen_modes_smoke_v1 quality metrics','formal477_review_initial_panels aspect ratio'],
        'failures_and_invalidated_outputs_preserved':True,'no_formal_retuning':True}
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','invalidated_early_outputs')},indent=2))


if __name__=='__main__':main()
