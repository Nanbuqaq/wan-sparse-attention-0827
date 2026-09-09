import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from adapters.longlive_sparse.object_state_protocol import validate_object_state_screen
from scripts.run_longlive2_native_reference import native_cut_schedule

ROOT=Path(__file__).resolve().parents[1]


def test_only_return_text_changes_and_cut_roles_remain_fixed():
    segments,base=native_cut_schedule(ROOT,'chest_revisit')
    changed,actual=native_cut_schedule(ROOT,'chest_revisit',object_text_control='past_settled_restatement')
    assert actual[0][:12]==base[0][:12] and changed[:-1]==segments[:-1]
    settled=next(s['prompt'] for s in segments if s['role']=='settled_source')
    assert all(actual[0][i]==base[0][i]+' '+settled for i in range(12,16))
    assert [i for i,p in enumerate(actual[0]) if p.startswith('The scene transitions. ')]==[2,6,12]
    assert changed[-1]['start_latent']==96 and segments[-1]['role']=='return_without_restatement'


def test_text_control_cannot_mix_with_KV_memory_or_envelope():
    args=SimpleNamespace(cut_scenario='chest_revisit',seed=20260925,object_state_text_control='past_settled_restatement')
    record=validate_object_state_screen(args,ROOT)
    assert record['Dense_only'] and record['text_control'] and not record['formal_holdout']
    for name,value in [('causal_scene_memory',True),('object_state_memory_study',True),('cut_scenario','envelope_revisit'),('seed',20260927)]:
        bad=SimpleNamespace(**vars(args));setattr(bad,name,value)
        with pytest.raises(ValueError):validate_object_state_screen(bad,ROOT)


@pytest.mark.parametrize('seed',[20260925,20260926])
def test_actual_text_control_CLI_preflight_does_not_allocate_or_write(tmp_path,seed):
    output=tmp_path/'not_created'
    command=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),'--assets',str(tmp_path/'unused'),
        '--output',str(output),'--object-protocol-only','--cut-scenario','chest_revisit','--seed',str(seed),
        '--native-local-frames','32','--cfg1-positive-cache-only','--fixed-adaln-warps','16','--fixed-adaln-stages','1',
        '--object-state-text-control','past_settled_restatement']
    result=subprocess.run(command,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    d=json.loads(result.stdout);assert d['blocks']==16 and d['scene_cuts']==[2,6,12]
    assert d['protocol']['Dense_only'] and d['protocol']['text_control']=='past_settled_restatement'
    assert not output.exists()
