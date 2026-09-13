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


def serial_task_groups(cases):
    """Keep each full task's alternating sequence together on a local pair."""
    references={c['id']:cases[c['reference_case_index']]['id'] for c in cases if 'reference_case_index' in c}
    ordered=[dict(c) for c in sorted(cases,key=lambda c:c.get('cohort_pair',0))]
    indices={c['id']:i for i,c in enumerate(ordered)}
    for c in ordered:
        c['cohort_task_index']=c.get('cohort_pair',0);c['cohort_pair']=0
        if c['id'] in references:c['reference_case_index']=indices[references[c['id']]]
    return ordered


def build_wave2_cases(spec,stage,assets,source,output,seed,valid_scenarios=None,expected_noise=None):
    if stage in ('matched_controls','timing_repeats','recall_toy','recall_bead','recall_replication','scene_release'):
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
    p.add_argument('--wave2-stage',choices=('native','algorithms','query_balance','matched_controls','timing_repeats','recall_toy','recall_bead','recall_replication','scene_release'),default='native')
    p.add_argument('--wave2-valid-scenarios',nargs='+')
    p.add_argument('--wave2-expected-noise')
    p.add_argument('--serial-task-groups',action='store_true',help='local fallback: whole task groups sequentially on one pair')
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
    pairs=args.gpu_pairs or min(len(cases),4)
    if args.serial_task_groups:
        if args.wave2_stage not in ('matched_controls','timing_repeats','scene_release') or pairs!=1:raise ValueError('serial fallback requires a complete matched cohort on one pair')
        cases=serial_task_groups(cases)
    if args.wave2_stage in ('matched_controls','timing_repeats') and pairs!=2 and not args.serial_task_groups:raise ValueError('matched controls require one physical pair per task')
    if args.wave2_stage in ('recall_toy','recall_bead') and pairs!=1:raise ValueError('recall factorial stays on one physical pair')
    if args.wave2_stage=='recall_replication' and pairs!=2:raise ValueError('second-seed regression requires one pair per task')
    if args.wave2_stage=='scene_release' and pairs!=2 and not args.serial_task_groups:raise ValueError('each scene-control task requires its own complete pair')
    if pairs>len(cases):raise ValueError('every GPU pair must have real cases')
    plan=dict(code_sha=sha,cases=cases,latent_frames=args.latent_frames,seed=args.seed,
        requested_GPU_count=2*pairs,two_GPUs_charged_per_case=True,noise_alignment=args.noise_alignment,
        lane_cases=[[c['id'] for c in cases[i::pairs]] for i in range(pairs)],
        scope=('Wave2 temporal budget factorial; no geometry model, per-task native review before algorithm stage'
            if args.wave2_config else 'geometry platform qualification; strict local-reference equality may fail across hardware, retain artifacts'
            if args.geometry_wave else 'registered duration or common-preparation comparison: fixed native32 and scripted real generated history'),
        CPU_review_runs_after_recovery=True)
    plan['serial_task_groups']=args.serial_task_groups
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
        if spec['latent_frames']!=128:raise ValueError('automatic noise gate freezes native128 geometry')
        check_noise='''import torch,json,hashlib,sys
torch.set_num_threads(2)
rows=[]
for device in range(0,torch.cuda.device_count(),2):
 torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 x=torch.randn(1,128,48,44,80,device=f'cuda:{device}',dtype=torch.bfloat16).cpu().contiguous()
 h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape)).encode());h.update(x.view(torch.uint8).numpy().tobytes())
 rows.append(dict(device=device,GPU=torch.cuda.get_device_name(device),noise_sha256=h.hexdigest()))
assert len({r['noise_sha256'] for r in rows})==1, 'lane initial noise differs'
print(json.dumps(dict(shape=[1,128,48,44,80],seed=SEED,rows=rows)))
'''.replace('SEED',str(args.seed))
        frozen_noise=json.loads(subprocess.check_output([sys.executable,'-c',check_noise],text=True))
        digest=frozen_noise['rows'][0]['noise_sha256']
        for case in cases:case['cmd']+=['--expected-noise-sha256',digest]
        (args.output/'input_noise_gate.json').write_text(json.dumps(frozen_noise,indent=2)+'\n')
        plan['actual_cohort_expected_noise']=digest
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
        for case_index in range(index,len(cases),pairs):
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
    if args.wave2_stage in ('matched_controls','timing_repeats'):
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
        system_equivalence_ok=len(comparisons)==2 and all(c['status']=='pass' for c in comparisons)
    terminal=dict(code_sha=sha,rows=rows,status='pass' if all(r['status']=='pass' and r['returncode']==0 for r in rows) else 'fail',
        semantic_review_complete=False,paired_prefix_review_pending=True)
    if not system_equivalence_ok:terminal.update(status='fail',same_route_equivalence_failed=True)
    (args.output/'batch_terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    if terminal['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
