from pathlib import Path
import pytest
from scripts.run_native_duration_wave import build_duration_cases
import json


def test_matched_noise_four_case_batch_has_unique_identity_and_two_method_controls():
    cases=build_duration_cases(scenarios=['generated_patchwork_toy_cut_revisit'],lengths=[128,728],seed=20261002,
        alignment='return_event',assets=Path('/assets'),source=Path('/source'),output=Path('/out'))
    assert len(cases)==4 and len({c['id'] for c in cases})==4
    assert {(c['latent_frames'],c['method']) for c in cases}=={(n,m) for n in (128,728) for m in ('native','scene_full')}
    for c in cases:
        assert '--duration-noise-alignment' in c['cmd']
        assert ('--causal-scene-memory' in c['cmd'])==(c['method']=='scene_full')
    with pytest.raises(ValueError):
        build_duration_cases(scenarios=['x'],lengths=[128,128],seed=1,alignment='absolute',assets=Path('/a'),source=Path('/s'),output=Path('/o'))


def test_long_condition_storage_pair_changes_only_preparation_and_output_identity():
    kwargs=dict(scenarios=['generated_patchwork_toy_cut_revisit'],lengths=[3608],seed=20261002,
        alignment='return_event',assets=Path('/assets'),source=Path('/source'),output=Path('/out'))
    a,b=build_duration_cases(**kwargs,methods=['native','native_shared'])
    assert a['id']!=b['id'] and '--native-shared-conditioning' in b['cmd']
    def normalize(cmd):
        cmd=cmd[:];i=cmd.index('--output');cmd[i+1]='OUTPUT'
        return [x for x in cmd if x!='--native-shared-conditioning']
    assert normalize(a['cmd'])==normalize(b['cmd'])
    assert all('--causal-scene-memory' not in c['cmd'] for c in (a,b))
    for bad in ([],['native','native'],['unknown']):
        with pytest.raises(ValueError):build_duration_cases(**kwargs,methods=bad)


def test_one_failed_case_does_not_repeat_or_block_other_cases(tmp_path,monkeypatch):
    import scripts.run_native_duration_wave as runner
    output=tmp_path/'batch';calls=[];components=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    monkeypatch.setattr(runner.sys,'argv',['run','--assets',str(tmp_path/'assets'),'--output',str(output),
        '--latent-frames','128','728','--seed','17','--scenario','generated_patchwork_toy_cut_revisit',
        '--noise-alignment','return_event','--gpu-pairs','2','--run'])
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:'f'*40+'\n')
    def execute(command,**kwargs):
        if any(str(x).endswith('gate_native_resident_component.py') for x in command):
            components.append(kwargs['env']['CUDA_VISIBLE_DEVICES']);return 0
        if any(str(x).endswith('run_longlive2_native_reference.py') for x in command):
            path=Path(command[command.index('--output')+1]);path.mkdir(parents=True,exist_ok=False)
            length=int(command[command.index('--duration-probe-latents')+1]);method='scene_full' if '--causal-scene-memory' in command else 'native'
            fail=length==128 and method=='native';calls.append((length,method,kwargs['env']['CUDA_VISIBLE_DEVICES']))
            (path/'summary.json').write_text(json.dumps({'status':'fail' if fail else 'pass'}));return int(fail)
        return 0
    monkeypatch.setattr(runner.subprocess,'call',execute)
    with pytest.raises(SystemExit) as error:runner.main()
    assert error.value.code==1 and len(calls)==len(set(calls))==4
    assert sorted(components)==['0,1','2,3']
    terminal=json.loads((output/'batch_terminal.json').read_text())
    assert sum(r['status']=='pass' for r in terminal['rows'])==3
    assert {(n,m) for n,m,_ in calls}=={(n,m) for n in (128,728) for m in ('native','scene_full')}
def test_geometry_wave_preserves_live_input_and_noise_boundaries(tmp_path):
    from scripts.run_native_duration_wave import build_geometry_cases
    cases=build_geometry_cases(assets=tmp_path/'assets',source=tmp_path/'source',output=tmp_path/'out')
    assert len(cases)==len({c['id'] for c in cases})==4
    for case in cases:
        command=case['cmd']
        assert '--duration-probe-latents' not in command
        assert '--duration-noise-alignment' not in command
        assert '--source-mask-oracle' not in command
        assert '--native-shared-conditioning' in command
        if case['method'].startswith('geometry_'):
            assert '--live-source-geometry' in command and '--equivalence-reference' in command
        else:
            assert '--geometry-checkpoint' not in command
    hold=next(c['cmd'] for c in cases if c['method']=='geometry_holdfirst')
    assert hold[hold.index('--geometry-mask-stride')+1]=='32'
def test_shared_geometry_reference_refuses_different_source_masks(tmp_path):
    import torch
    import pytest
    from scripts.run_native_duration_wave import build_geometry_reference_cases
    for name,ids in [('all32',[1,2]),('holdfirst',[1,3])]:
        folder=tmp_path/name;folder.mkdir()
        torch.save({'indices':torch.tensor(ids),'source_latent_sha256':'same'},folder/'source_mask_indices.pt')
    with pytest.raises(ValueError,match='identical'):
        build_geometry_reference_cases(assets=tmp_path,source=tmp_path,output=tmp_path/'out',geometry_inputs=tmp_path,source_masks=tmp_path)
    torch.save({'indices':torch.tensor([1,2]),'source_latent_sha256':'same'},tmp_path/'holdfirst/source_mask_indices.pt')
    cases=build_geometry_reference_cases(assets=tmp_path,source=tmp_path,output=tmp_path/'out',geometry_inputs=tmp_path,source_masks=tmp_path)
    assert all(c['reference_case_index']<i for i,c in enumerate(cases) if 'reference_case_index' in c)
    assert all(c['cmd'][c['cmd'].index('--source-mask-fill')+1]=='fixed_bit_reversal' for c in cases[2:])
