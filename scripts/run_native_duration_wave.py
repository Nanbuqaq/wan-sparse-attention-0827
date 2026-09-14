#!/usr/bin/env python3
"""Assign two GPUs to each independent native/full-source duration case."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SCENARIOS=('generated_patchwork_toy_cut_revisit','generated_bead_state_cut_revisit')


def case_lane_indices(cases,pairs):
    lanes=[[] for _ in range(pairs)]
    for i,case in enumerate(cases):
        lane=case.get('cohort_pair',i%pairs)
        if not 0<=lane<pairs:raise ValueError('case requests unavailable physical pair')
        lanes[lane].append(i)
    if any(not lane for lane in lanes):raise ValueError('every pair requires actual cases')
    return lanes


def serial_task_groups(cases):
    """Keep each full task's alternating sequence together on a local pair."""
    references={c['id']:cases[c['reference_case_index']]['id'] for c in cases if 'reference_case_index' in c}
    ordered=[dict(c) for c in sorted(cases,key=lambda c:c.get('cohort_pair',0))]
    indices={c['id']:i for i,c in enumerate(ordered)}
    for c in ordered:
        c['cohort_task_index']=c.get('cohort_pair',0);c['cohort_pair']=0
        if c['id'] in references:c['reference_case_index']=indices[references[c['id']]]
    return ordered


def with_common_inplace_gelu(cases,native_reference=None):
    updated=[]
    for case in cases:
        item=dict(case,cmd=list(case['cmd']))
        item['cmd']+=['--native-inplace-gelu'];item['common_native_inplace_gelu']=True
        updated.append(item)
    if native_reference is not None:
        if not updated or updated[0]['method']!='native':raise ValueError('common system reference must guard the first native case')
        updated[0]['cmd']+=['--equivalence-reference',str(native_reference)]
        updated[0]['guarded_native_equivalence']=True
    return updated


def build_wave2_cases(spec,stage,assets,source,output,seed,valid_scenarios=None,expected_noise=None):
    if stage=='archive_system':
        from scripts.archive_system_cohort import build_archive_system
        if expected_noise:raise ValueError('archive timing wave requires matched hardware input gates')
        return build_archive_system(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='conditional_delta':
        from scripts.conditional_delta_cohort import build_conditional_delta
        if expected_noise:raise ValueError('conditional wave requires per-context hardware input gates')
        return build_conditional_delta(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='source_representation':
        from scripts.source_representation_cohort import build_source_representation
        if expected_noise:raise ValueError('source representation requires its own hardware input gates')
        return build_source_representation(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='source_two':
        from scripts.source_two_cohort import build_source_two
        if expected_noise:raise ValueError('two-source-call study requires per-seed hardware input gates')
        return build_source_two(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='source_first':
        from scripts.source_first_cohort import build_source_first
        if expected_noise:raise ValueError('first-source study requires a hardware-specific input gate')
        return build_source_first(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='source_clean':
        from scripts.source_clean_cohort import build_source_clean
        if expected_noise:raise ValueError('clean study requires its own per-seed noise gates')
        return build_source_clean(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='state_representation':
        from scripts.state_representation_cohort import build_state_representation
        if expected_noise:raise ValueError('representation study requires its own hardware input gate')
        return build_state_representation(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='fused_query_system':
        from scripts.fused_query_cohort import build_fused_query
        if expected_noise:raise ValueError('backend batch requires its own hardware input gate')
        return build_fused_query(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='state_snapshot_replication':
        from scripts.state_snapshot_cohort import build_state_snapshot_replication
        if expected_noise:raise ValueError('snapshot replication needs per-seed noise gates')
        return build_state_snapshot_replication(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='state_snapshot':
        from scripts.state_snapshot_cohort import build_state_snapshot
        if expected_noise:raise ValueError('state snapshot needs its own per-seed noise gate')
        return build_state_snapshot(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='context_write':
        if seed!=20261010 or expected_noise:raise ValueError('combined context/write wave has fixed per-case seeds')
        return (build_wave2_cases(spec,'return_context',assets,source,output,20261010)
                +build_wave2_cases(spec,'write_origin',assets,source,output,20260913))
    if stage=='return_context':
        from scripts.return_context_cohort import build_return_context
        if expected_noise:raise ValueError('return context uses its own per-seed noise gate')
        return build_return_context(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='write_origin':
        from scripts.write_origin_cohort import build_write_origin
        if expected_noise:raise ValueError('write cohort requires own per-seed noise gate')
        return build_write_origin(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='information_groups':
        from scripts.information_group_cohort import build_information_groups
        if expected_noise:raise ValueError('information groups require own per-seed noise gate')
        return build_information_groups(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='memory_mechanisms':
        if seed!=20261010 or expected_noise:raise ValueError('combined memory wave has fixed per-case seeds and its own noise gate')
        return (build_wave2_cases(spec,'source_lifetime',assets,source,output,20261010)
                +build_wave2_cases(spec,'state_feasibility',assets,source,output,20261021))
    if stage=='state_feasibility':
        from scripts.state_feasibility_cohort import build_state_feasibility
        if expected_noise:raise ValueError('state cohort requires own per-seed noise gate')
        return build_state_feasibility(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='source_lifetime':
        from scripts.source_lifetime_cohort import build_source_lifetime
        if expected_noise:raise ValueError('lifetime cohort requires own per-seed noise gate')
        return build_source_lifetime(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='query_groups':
        from scripts.query_group_cohort import build_query_groups
        if expected_noise:raise ValueError('query cohort requires its own per-seed noise gate')
        return build_query_groups(spec,assets,source,output,seed,build_wave2_cases)
    if stage=='source_layers':
        from scripts.source_layer_cohort import build_source_layers
        if expected_noise:raise ValueError('layer cohort requires own per-seed noise gate')
        return build_source_layers(spec,assets,source,output,seed,build_wave2_cases)
    if stage in ('multi_event','lineage_controls'):
        from scripts.multi_event_cohort import build_multi_event
        if expected_noise:raise ValueError('multi-event requires own per-seed noise gate')
        return build_multi_event(spec,assets,source,output,seed,build_wave2_cases,lineage=stage=='lineage_controls')
    if stage=='read_and_route':
        from scripts.read_route_cohort import build_read_route
        if expected_noise:raise ValueError('read/route cohort requires own per-seed noise gate')
        return build_read_route(spec,assets,source,output,seed,build_wave2_cases)
    if stage in ('motion_long','archive_timing','source_weight','delayed_and_return','context_controls'):
        from scripts.access_motion_followups import build_followup
        if expected_noise:raise ValueError('followup needs own per-seed noise gate')
        return build_followup(spec,stage,assets,source,output,seed,build_wave2_cases)
    if stage=='semantic_versions':
        from scripts.semantic_version_cohort import build_semantic_cohort
        if expected_noise:raise ValueError('semantic wave needs own noise gate')
        return build_semantic_cohort(spec,assets,source,output,seed,build_wave2_cases)
    if stage in ('access_motion_first','access_factorial'):
        from scripts.access_motion_cohort import build_access_cohort
        if expected_noise:raise ValueError('new cohort requires own noise preflight')
        return build_access_cohort(spec,stage,assets,source,output,seed,build_wave2_cases)
    if stage in ('matched_controls','timing_repeats','recall_toy','recall_bead','recall_replication','scene_release','recent_control','recent_hopper_control','long_sum_regression','long_quality_replication'):
        from scripts.next24h_cohort import build_cohort
        if expected_noise:raise ValueError('new homogeneous cohort requires its own noise preflight')
        return build_cohort(spec,stage,assets,source,output,seed,build_wave2_cases)
    if seed not in (spec['development_seed'],spec['replication_seed']):raise ValueError('unregistered Wave2 seed')
    if stage!='native' and valid_scenarios is None:raise ValueError('native task-validity review is required')
    if stage=='query_balance':
        valid=set(valid_scenarios)
        if not valid or not valid<=set(('w2_rotating_wooden_bird','w2_tracking_delivery_cart')):
            raise ValueError('P3 first wave freezes the two valid continuous tasks')
        base=build_wave2_cases(spec,'algorithms',assets,source,output,seed,valid_scenarios,expected_noise)
        cases=[]
        for row in base:
            for selector in ('query_sum_batch4','query_balanced_batch4'):
                name=row['id']+'__'+selector;cmd=list(row['cmd'])
                if '--wave2-capture' in cmd:cmd.remove('--wave2-capture')
                cmd[cmd.index('--output')+1]=str(output/name)
                cmd+=['--wave2-selector',selector]
                cases.append(dict(row,id=name,cmd=cmd,selector=selector))
        return cases
    cases=[]
    scenarios=spec['scenarios']+(spec.get('backup_scenarios',[]) if valid_scenarios is not None else [])
    for scenario in scenarios:
        if valid_scenarios is not None and scenario['id'] not in valid_scenarios:continue
        for method in scenario['methods']:
            if (method=='w2_native')!=(stage=='native'):continue
            name=f"{scenario['id']}__s{seed}__{method}"
            cmd=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),
                '--assets',str(assets),'--source',str(source),'--output',str(output/name),
                '--cut-scenario',scenario['id'],'--seed',str(seed),'--wave2-method',method,
                '--native-local-frames','32','--cfg1-positive-cache-only','--native-inplace-cache','--native-shared-conditioning',
                '--fixed-adaln-warps','16','--fixed-adaln-stages','1','--constructor-mode','strict_checkpoint_no_parameter_init',
                '--pipeline-mode','overlap','--pipeline-encode-mode','thread','--wave2-steady-fraction',str(spec['steady_fraction'])]
            if method=='w2_steady_sparse' and scenario['id'] in ('w2_rotating_wooden_bird','w2_tracking_delivery_cart','w2_ceramic_jug_revisit','w2_settled_pebble_bowl_backup'):
                cmd+=['--wave2-capture']
            if expected_noise:cmd+=['--expected-noise-sha256',expected_noise]
            cases.append(dict(id=name,scenario=scenario['id'],method=method,latent_frames=spec['latent_frames'],cmd=cmd))
    return cases


def build_geometry_cases(*,assets,source,output,geometry_inputs=None):
    cases=[]
    for method in ('native','scene_full','geometry_all32','geometry_holdfirst'):
        case=build_duration_cases(scenarios=(SCENARIOS[0],),lengths=(128,),seed=20260913,
            alignment='absolute',assets=assets,source=source,output=output,methods=('native',))[0]
        cmd=case['cmd']
        for option in ('--duration-probe-latents','--duration-noise-alignment'):
            i=cmd.index(option);del cmd[i:i+2]
        name='toy13__'+method;cmd[cmd.index('--output')+1]=str(output/name)
        cmd+=['--native-shared-conditioning']
        if method=='scene_full':cmd+=['--causal-block-policy','full','--causal-block-fraction','1']
        if method.startswith('geometry_'):
            inputs=geometry_inputs or assets/'geometry_wave_v1'
            cmd+=['--causal-block-policy','source_mask','--causal-block-fraction','.25',
                '--live-source-geometry','--geometry-compact-return',
                '--geometry-checkpoint',str(inputs/'sam2_hiera_large.pt'),
                '--equivalence-reference',str(inputs/(method+'_reference.json'))]
            if method=='geometry_holdfirst':cmd+=['--geometry-mask-stride','32','--source-mask-fill','fixed_bit_reversal']
        case.update(id=name,method=method,cmd=cmd)
        cases.append(case)
    return cases


def build_geometry_reference_cases(*,assets,source,output,geometry_inputs,source_masks):
    import torch
    a=torch.load(source_masks/'all32/source_mask_indices.pt',weights_only=True,map_location='cpu')
    b=torch.load(source_masks/'holdfirst/source_mask_indices.pt',weights_only=True,map_location='cpu')
    if not torch.equal(a['indices'],b['indices']) or a['source_latent_sha256']!=b['source_latent_sha256']:
        raise ValueError('shared-reference wave requires exactly identical foreground source indices')
    if not 0<a['indices'].numel()<=1760:raise ValueError('source masks must fit the registered budget')
    base=build_geometry_cases(assets=assets,source=source,output=output,geometry_inputs=geometry_inputs)
    cases=[]
    for fill in ('uniform_midpoint','fixed_bit_reversal'):
        method='reference_uniform' if fill=='uniform_midpoint' else 'reference_stable'
        name='toy13__'+method;case=dict(base[0]);cmd=list(case['cmd']);cmd[cmd.index('--output')+1]=str(output/name)
        cmd+=['--causal-block-policy','source_mask','--causal-block-fraction','.25',
            '--source-mask-oracle',str(source_masks/'all32/source_mask_indices.pt'),'--source-mask-fill',fill]
        case.update(id=name,method=method,cmd=cmd);cases.append(case)
    for case in base[2:]:
        cmd=case['cmd'];cmd[cmd.index('--equivalence-reference')+1]=str(output/'toy13__reference_stable/summary.json')
        if '--source-mask-fill' not in cmd:cmd+=['--source-mask-fill','fixed_bit_reversal']
        case['reference_case_index']=1;cases.append(case)
    return cases


def build_duration_cases(*,scenarios,lengths,seed,alignment,assets,source,output,methods=('native','scene_full')):
    if not lengths or len(lengths)!=len(set(lengths)) or any(x not in (128,184,728,3608) for x in lengths):
        raise ValueError('distinct registered durations required')
    if alignment not in ('absolute','return_event'):raise ValueError('unknown noise alignment')
    if not methods or len(methods)!=len(set(methods)) or set(methods)-{'native','scene_full','native_shared'}:
        raise ValueError('distinct registered duration methods required')
    cases=[]
    for scenario in scenarios:
        for length in lengths:
            for method in methods:
                name=f'{scenario}__s{seed}__T{length}__noise_{alignment}__{method}'
                cmd=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),
                    '--assets',str(assets),'--source',str(source),'--output',str(output/name),
                    '--cut-scenario',scenario,'--seed',str(seed),'--duration-probe-latents',str(length),
                    '--duration-noise-alignment',alignment,
                    '--native-local-frames','32','--cfg1-positive-cache-only','--native-inplace-cache',
                    '--fixed-adaln-warps','16','--fixed-adaln-stages','1',
                    '--constructor-mode','strict_checkpoint_no_parameter_init',
                    '--pipeline-mode','overlap','--pipeline-encode-mode','thread']
                if method=='scene_full':cmd+=['--causal-scene-memory']
                if method=='native_shared':cmd+=['--native-shared-conditioning']
                cases.append(dict(id=name,scenario=scenario,method=method,latent_frames=length,noise_alignment=alignment,cmd=cmd))
    return cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'third_party/LongLive2');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--latent-frames',type=int,nargs='+',choices=(128,184,728,3608));p.add_argument('--seed',type=int,required=True)
    p.add_argument('--wave2-config',type=Path)
    p.add_argument('--wave2-stage',choices=('native','algorithms','query_balance','matched_controls','timing_repeats','recall_toy','recall_bead','recall_replication','scene_release','recent_control','recent_hopper_control','long_sum_regression','long_quality_replication','access_motion_first','access_factorial','semantic_versions','motion_long','archive_timing','source_weight','delayed_and_return','read_and_route','context_controls','multi_event','source_layers','lineage_controls','query_groups','source_lifetime','state_feasibility','memory_mechanisms','information_groups','write_origin','return_context','context_write','state_snapshot','state_snapshot_replication','fused_query_system','state_representation','source_clean','source_first','source_two','source_representation','conditional_delta','archive_system'),default='native')
    p.add_argument('--wave2-valid-scenarios',nargs='+')
    p.add_argument('--wave2-expected-noise')
    p.add_argument('--serial-task-groups',action='store_true',help='local fallback: whole task groups sequentially on one pair')
    p.add_argument('--native-inplace-gelu',action='store_true',help='same native FFN buffer optimization for every case')
    p.add_argument('--native-equivalence-reference',type=Path,help='guard the first native system-change case before the rest of its lane')
    p.add_argument('--stop-lane-on-oom',action='store_true',help='preserve first capacity failure and skip dependent local repetitions')
    p.add_argument('--noise-alignment',choices=('absolute','return_event'),default='absolute')
    p.add_argument('--methods',nargs='+',choices=('native','scene_full','native_shared'),default=('native','scene_full'))
    p.add_argument('--geometry-wave',action='store_true',help='frozen toy13 native/full/live geometry platform qualification')
    p.add_argument('--geometry-inputs',type=Path)
    p.add_argument('--geometry-recovery-only',action='store_true')
    p.add_argument('--geometry-reference-wave',action='store_true')
    p.add_argument('--geometry-source-masks',type=Path)
    p.add_argument('--gpu-pairs',type=int,choices=(1,2,4),help='reuse each assigned pair for its sequential cases')
    p.add_argument('--scenario',choices=(*SCENARIOS,'both'));p.add_argument('--run',action='store_true')
    p.add_argument('--required-gpu-name',default='H200');p.add_argument('--allow-h800',action='store_true');args=p.parse_args()
    scenarios=SCENARIOS if args.scenario=='both' else (args.scenario,)
    visible=[x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if args.wave2_config:
        if args.geometry_wave or args.geometry_recovery_only or args.geometry_reference_wave:raise ValueError('Wave2 cannot use old geometry waves')
        spec=json.loads(args.wave2_config.read_text());args.latent_frames=[spec['latent_frames']]
        cases=build_wave2_cases(spec,args.wave2_stage,args.assets,args.source,args.output,args.seed,args.wave2_valid_scenarios,args.wave2_expected_noise)
        if not cases:raise ValueError('no valid new cases; do not reserve GPUs')
        args.latent_frames=sorted({c['latent_frames'] for c in cases})
    elif args.geometry_wave:
        if args.latent_frames!=[128] or args.seed!=20260913 or scenarios!=(SCENARIOS[0],) or args.noise_alignment!='absolute':
            raise ValueError('geometry qualification is the frozen toy13 absolute-noise full509 slice')
        if args.geometry_reference_wave:
            if args.geometry_recovery_only or args.geometry_source_masks is None:
                raise ValueError('reference wave needs qualified source masks and all four cases')
            cases=build_geometry_reference_cases(assets=args.assets,source=args.source,output=args.output,
                geometry_inputs=args.geometry_inputs,source_masks=args.geometry_source_masks)
        else:
            cases=build_geometry_cases(assets=args.assets,source=args.source,output=args.output,geometry_inputs=args.geometry_inputs)
            if args.geometry_recovery_only:cases=[c for c in cases if c['method'].startswith('geometry_')]
    else:
        if args.geometry_recovery_only or args.geometry_reference_wave:raise ValueError('geometry recovery requires its registered wave')
        if args.scenario is None or args.latent_frames is None:raise ValueError('duration scenario and length required')
        cases=build_duration_cases(scenarios=scenarios,lengths=args.latent_frames,seed=args.seed,alignment=args.noise_alignment,
            assets=args.assets,source=args.source,output=args.output,methods=args.methods)
    if args.native_inplace_gelu:cases=with_common_inplace_gelu(cases,args.native_equivalence_reference)
    elif args.native_equivalence_reference:raise ValueError('native system reference requires its common optimization')
    pairs=args.gpu_pairs or min(len(cases),4)
    if args.serial_task_groups:
        if args.wave2_stage not in ('matched_controls','timing_repeats','scene_release','recent_control','information_groups') or pairs!=1:raise ValueError('serial fallback requires a complete matched cohort on one pair')
        cases=serial_task_groups(cases)
    if args.wave2_stage in ('matched_controls','timing_repeats') and pairs!=2 and not args.serial_task_groups:raise ValueError('matched controls require one physical pair per task')
    if args.wave2_stage in ('recall_toy','recall_bead') and pairs!=1:raise ValueError('recall factorial stays on one physical pair')
    if args.wave2_stage=='recall_replication' and pairs!=2:raise ValueError('second-seed regression requires one pair per task')
    if args.wave2_stage=='scene_release' and pairs!=2 and not args.serial_task_groups:raise ValueError('each scene-control task requires its own complete pair')
    if args.wave2_stage=='recent_control' and pairs!=2 and not args.serial_task_groups:raise ValueError('recent control requires complete native pairing')
    if args.wave2_stage=='recent_hopper_control' and pairs!=2:raise ValueError('complete recent comparison requires one pair per task')
    if args.wave2_stage=='long_sum_regression' and pairs!=1:raise ValueError('long regression stays on one pair')
    if args.wave2_stage=='long_quality_replication' and pairs!=1:raise ValueError('quality replication stays on one pair')
    if args.wave2_stage=='access_motion_first' and pairs!=2:raise ValueError('first access/motion stage keeps complete task groups on two balanced pairs')
    if args.wave2_stage=='access_factorial' and pairs!=2:raise ValueError('access factorial has two complete task pairs')
    if args.wave2_stage=='semantic_versions' and pairs!=2:raise ValueError('semantic wave requires two balanced complete task pairs')
    if args.wave2_stage in ('motion_long','archive_timing','source_weight','delayed_and_return','context_controls') and pairs!=2:raise ValueError('registered followup needs two complete pairs')
    if args.wave2_stage=='read_and_route' and pairs!=2:raise ValueError('read/route cohort has two balanced complete pairs')
    if args.wave2_stage in ('multi_event','lineage_controls') and pairs!=2:raise ValueError('multi-event keeps each seed on one pair')
    if args.wave2_stage=='source_layers' and pairs!=2:raise ValueError('layer controls keep each task on one pair')
    if args.wave2_stage=='query_groups' and pairs!=2:raise ValueError('query groups keep each complete task on one pair')
    if args.wave2_stage=='source_lifetime' and pairs!=2:raise ValueError('source lifetime keeps each task on one pair')
    if args.wave2_stage=='state_feasibility' and pairs!=2:raise ValueError('state feasibility keeps each task on one pair')
    if args.wave2_stage=='memory_mechanisms' and pairs!=2:raise ValueError('balanced source/state wave uses two complete physical pairs')
    if args.wave2_stage=='information_groups' and pairs!=2 and not args.serial_task_groups:raise ValueError('information groups keep each full task on one pair')
    if args.wave2_stage=='write_origin' and pairs!=2:raise ValueError('write origin keeps each task on one physical pair')
    if args.wave2_stage=='return_context' and pairs!=2:raise ValueError('return context keeps each full task on one physical pair')
    if args.wave2_stage=='context_write' and pairs!=2:raise ValueError('combined context/write wave uses two balanced physical pairs')
    if args.wave2_stage=='state_snapshot' and pairs!=2:raise ValueError('state snapshot keeps each request and its controls on one pair')
    if args.wave2_stage=='state_snapshot_replication' and pairs!=2:raise ValueError('snapshot replication keeps each request and controls on one pair')
    if args.wave2_stage=='fused_query_system' and pairs!=2:raise ValueError('backend study keeps both orders and controls on each task pair')
    if args.wave2_stage=='state_representation' and pairs!=2:raise ValueError('representation study keeps each request and controls on one pair')
    if args.wave2_stage=='source_clean' and pairs!=4:raise ValueError('clean study has four matched source/request contexts')
    if args.wave2_stage=='source_first' and pairs!=2:raise ValueError('single-source-call study keeps each request on one pair')
    if args.wave2_stage=='source_two' and pairs!=4:raise ValueError('two-call study has four matched state/detail contexts')
    if pairs>len(cases):raise ValueError('every GPU pair must have real cases')
    lane_indices=case_lane_indices(cases,pairs)
    plan=dict(code_sha=sha,cases=cases,latent_frames=args.latent_frames,seed=args.seed,
        requested_GPU_count=2*pairs,two_GPUs_charged_per_case=True,noise_alignment=args.noise_alignment,
        lane_cases=[[cases[j]['id'] for j in lane] for lane in lane_indices],
        scope=('Wave2 temporal budget factorial; no geometry model, per-task native review before algorithm stage'
            if args.wave2_config else 'geometry platform qualification; strict local-reference equality may fail across hardware, retain artifacts'
            if args.geometry_wave else 'registered duration or common-preparation comparison: fixed native32 and scripted real generated history'),
        CPU_review_runs_after_recovery=True)
    plan['serial_task_groups']=args.serial_task_groups
    plan['common_native_inplace_gelu']=args.native_inplace_gelu
    plan['native_equivalence_reference']=str(args.native_equivalence_reference) if args.native_equivalence_reference else None
    plan['case_seeds']=sorted({int(c['cmd'][c['cmd'].index('--seed')+1]) for c in cases})
    if args.wave2_config:
        plan['wave2_config_sha256']=hashlib.sha256(args.wave2_config.read_bytes()).hexdigest()
        plan['wave2_stage']=args.wave2_stage;plan['valid_scenarios']=args.wave2_valid_scenarios
    if not args.run:print(json.dumps(plan,indent=2));return
    if len(visible)!=2*pairs or len(visible)!=len(set(visible)):
        raise ValueError('exactly two distinct assigned GPUs per real case required')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'batch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    topology=[sys.executable,str(ROOT/'scripts/check_native_hardware.py'),'--expected-count',str(len(visible)),
        '--required',args.required_gpu_name,'--output',str(args.output/'hardware.json')]
    if args.allow_h800:topology+=['--allow-h800']
    if subprocess.call(topology):
        (args.output/'batch_terminal.json').write_text(json.dumps(dict(code_sha=sha,status='fail',
            rows=[dict(id=c['id'],status='blocked_by_hardware_topology',returncode=1) for c in cases]))+'\n')
        raise SystemExit(1)
    if args.wave2_config and not args.wave2_expected_noise:
        # A new homogeneous-platform cohort verifies its actual initial noise
        # before any video, without borrowing a different GPU model's hash.
        lengths={c['latent_frames'] for c in cases}
        if len(lengths)!=1 or not lengths<={128,728}:raise ValueError('automatic noise gate requires registered homogeneous length')
        noise_length=next(iter(lengths))
        case_seed=lambda c:int(c['cmd'][c['cmd'].index('--seed')+1])
        seeds=sorted({case_seed(c) for c in cases})
        frozen_noise=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/probe_cohort_noise.py'),
            '--length',str(noise_length),'--seeds',*map(str,seeds)],text=True))
        digests={row['seed']:row['noise_sha256'] for row in frozen_noise['rows']}
        for case in cases:case['cmd']+=['--expected-noise-sha256',digests[case_seed(case)]]
        (args.output/'input_noise_gate.json').write_text(json.dumps(frozen_noise,indent=2)+'\n')
        plan['actual_cohort_expected_noise']=digests[seeds[0]] if len(seeds)==1 else None
        plan['actual_expected_noise_by_seed']=digests
        (args.output/'batch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    accepted=(args.required_gpu_name,'H800') if args.allow_h800 else (args.required_gpu_name,)
    def run_lane(index):
        env=os.environ.copy();devices=','.join(visible[2*index:2*index+2])
        env.update(CUDA_VISIBLE_DEVICES=devices,WAN_SPARSE_PHYSICAL_GPUS=devices,
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',
            LLV2_USE_FA3='0',LLV2_USE_FA4='0',LLV2_USE_TE_ATTN='0',LLV2_COMPILE_VAE='0',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        env.pop('WAN_SPARSE_PHYSICAL_GPU',None)
        check='import torch; assert torch.cuda.device_count()==2; names=[torch.cuda.get_device_name(i) for i in range(2)]; print(names); assert all(any(x in n for x in '+repr(accepted)+') for n in names)'
        gate=1;gate_kind='hardware';gate_error=None
        try:
            with (args.output/f'lane{index}_hardware.log').open('x') as handle:
                gate=subprocess.call([sys.executable,'-c',check],env=env,stdout=handle,stderr=subprocess.STDOUT)
            gate_kind='hardware'
            if not gate:
                gate_kind='component'
                with (args.output/f'lane{index}_component.log').open('x') as handle:
                    gate=subprocess.call([sys.executable,str(ROOT/'scripts/gate_native_resident_component.py'),
                        '--output',str(args.output/f'component_lane{index}')],env=env,stdout=handle,stderr=subprocess.STDOUT)
        except Exception as error:gate=-1;gate_error=repr(error)
        lane_rows=[]
        for case_index in lane_indices[index]:
            case=cases[case_index]
            row=dict(id=case['id'],scenario=case['scenario'],method=case['method'],lane=index,case_index=case_index,assigned_devices=devices)
            dependency=case.get('reference_case_index');dependency_ok=True
            if dependency is not None and not gate:
                if dependency>=case_index:raise ValueError('references must precede dependent cases')
                reference=args.output/f'case{dependency}_terminal.json';deadline=time.monotonic()+300
                while not reference.exists() and time.monotonic()<deadline:time.sleep(.1)
                dependency_ok=reference.exists() and json.loads(reference.read_text()).get('status')=='pass'
            if gate:row.update(status='blocked_by_'+gate_kind+'_gate',returncode=gate,error=gate_error)
            elif not dependency_ok:row.update(status='blocked_by_reference_gate',returncode=1)
            else:
                try:
                    with (args.output/(case['id']+'.log')).open('x') as handle:
                        code=subprocess.call(case['cmd'],env=env,stdout=handle,stderr=subprocess.STDOUT)
                    path=args.output/case['id']/'summary.json';d=json.loads(path.read_text()) if path.exists() else {}
                    row.update(status=d.get('status','missing') if code==0 else 'fail',returncode=code,summary=str(path))
                    if case.get('guarded_native_equivalence') and row['status']!='pass':
                        gate=1;gate_kind='native_equivalence';gate_error='common system change failed native output equivalence; dependent cases not run'
                    if args.stop_lane_on_oom and code and 'out of memory' in d.get('traceback','').lower():
                        gate=1;gate_kind='capacity';gate_error='prior case OOM; no repeated same-geometry local attempts'
                except Exception as error:row.update(status='fail',returncode=-1,error=repr(error))
            terminal_path=args.output/f'case{case_index}_terminal.json'
            temporary=terminal_path.with_suffix('.json.tmp');temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(terminal_path)
            lane_rows.append(row);print(json.dumps(row),flush=True)
        (args.output/f'lane{index}_terminal.json').write_text(json.dumps(lane_rows,indent=2)+'\n')
        return lane_rows
    with ThreadPoolExecutor(max_workers=pairs) as pool:rows=[r for group in pool.map(run_lane,range(pairs)) for r in group]
    system_equivalence_ok=True
    if args.wave2_stage in ('matched_controls','timing_repeats','long_sum_regression'):
        comparisons=[]
        for scenario in {c['scenario'] for c in cases}:
            selected=[c for c in cases if c['scenario']==scenario and c['method'] in ('sum_old','sum_fast','sum_observer')]
            if all((args.output/c['id']/'summary.json').exists() for c in selected):
                reports=[json.loads((args.output/c['id']/'summary.json').read_text()) for c in selected]
                equal=all(d.get('status')=='pass' for d in reports) and all(len(set(values))==1 for values in (
                    [d.get('noise_sha256') for d in reports],[d.get('latent_sha256') for d in reports],
                    [d.get('pixels',{}).get('raw_RGB_sha256') for d in reports],
                    [d.get('wave2',{}).get('route_audit_sha256') for d in reports]))
                comparisons.append(dict(scenario=scenario,status='pass' if equal else 'fail',scope='complete noise/latent/rawRGB/compact route+binding equality'))
        (args.output/'same_route_equivalence.json').write_text(json.dumps(comparisons,indent=2)+'\n')
        system_equivalence_ok=len(comparisons)==len({c['scenario'] for c in cases}) and all(c['status']=='pass' for c in comparisons)
    terminal=dict(code_sha=sha,rows=rows,status='pass' if all(r['status']=='pass' and r['returncode']==0 for r in rows) else 'fail',
        semantic_review_complete=False,paired_prefix_review_pending=True)
    if not system_equivalence_ok:terminal.update(status='fail',same_route_equivalence_failed=True)
    (args.output/'batch_terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    if terminal['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
