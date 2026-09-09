from pathlib import Path
from types import SimpleNamespace

import pytest
import json
import subprocess
import sys

from adapters.longlive_sparse.object_state_protocol import SCENARIOS,validate_object_state_screen
from scripts.run_longlive2_native_reference import native_cut_schedule

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('obj',['chest','envelope'])
def test_revisit_and_visible_control_share_source_and_cut_boundaries(obj):
    segments,a=native_cut_schedule(ROOT,obj+'_revisit')
    _,b=native_cut_schedule(ROOT,obj+'_visible_control')
    assert a[0][:6]==b[0][:6] and len(a[0])==len(b[0])==16
    assert [i for i,p in enumerate(a[0]) if p.startswith('The scene transitions. ')]==[2,6,12]
    assert [i for i,p in enumerate(b[0]) if p.startswith('The scene transitions. ')]==[2,6,12]
    assert [s['start_latent'] for s in segments]==[0,16,32,48,96]
    assert segments[2]['role']=='settled_source' and segments[3]['role']=='away'
    assert 'Back to ' in a[0][12] and 'Stay with ' in b[0][12]


@pytest.mark.parametrize('scenario',SCENARIOS)
def test_state_source_feasibility_is_not_a_low_resolution_gate(scenario):
    with pytest.raises(ValueError):native_cut_schedule(ROOT,scenario,gate=True)


def test_screen_refuses_unfrozen_seed_or_memory_intervention():
    good=SimpleNamespace(cut_scenario='chest_revisit',seed=20260925)
    assert validate_object_state_screen(good,ROOT)['formal_holdout'] is False
    for key,value in [('seed',20260927),('causal_scene_memory',True),('episode_memory_mode','raw_reveal'),
                      ('pipeline_mode','overlap'),('capture_attention_teacher',True)]:
        bad=SimpleNamespace(**vars(good));setattr(bad,key,value)
        with pytest.raises(ValueError):validate_object_state_screen(bad,ROOT)


def test_only_reviewed_chest_can_run_the_frozen_causal_study():
    args=SimpleNamespace(cut_scenario='chest_revisit',seed=20260925,object_state_memory_study=True,
                         causal_scene_memory=True,causal_scene_position_policy='original')
    d=validate_object_state_screen(args,ROOT)
    assert not d['Dense_only'] and d['memory_study_registration']['selector']['margin']==.05
    args.cut_scenario='envelope_revisit'
    with pytest.raises(ValueError):validate_object_state_screen(args,ROOT)


@pytest.mark.parametrize('seed',[20260925,20260926])
@pytest.mark.parametrize('policy',['original','recent_virtual'])
def test_exact_full_batch_CLI_passes_all_object_and_causal_entry_guards(tmp_path,seed,policy):
    output=tmp_path/'must_not_exist'
    command=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),
        '--assets',str(tmp_path/'not_loaded'),'--output',str(output),'--object-protocol-only',
        '--cut-scenario','chest_revisit','--seed',str(seed),'--native-local-frames','32',
        '--cfg1-positive-cache-only','--fixed-adaln-warps','16','--fixed-adaln-stages','1',
        '--object-state-memory-study','--causal-scene-memory','--causal-scene-position-policy',policy]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    data=json.loads(result.stdout)
    assert data['status']=='pass' and data['latent_frames']==128 and data['scene_cuts']==[2,6,12]
    assert not data['model_or_GPU_initialized'] and not output.exists()
    command[command.index('chest_revisit')]='envelope_revisit'
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and not output.exists()
