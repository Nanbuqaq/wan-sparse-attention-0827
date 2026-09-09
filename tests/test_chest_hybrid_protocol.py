import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from adapters.longlive_sparse.object_state_protocol import validate_object_state_screen

ROOT=Path(__file__).resolve().parents[1]


def test_hybrid_is_one_explicit_cell_not_a_general_guard_relaxation():
    good=SimpleNamespace(cut_scenario='chest_revisit',seed=20260925,object_state_memory_study=True,
        object_state_text_control='past_settled_restatement',causal_scene_memory=True,
        causal_scene_position_policy='recent_virtual',chest_hybrid_study=True)
    assert validate_object_state_screen(good,ROOT)['hybrid_registration']['reused_factorial_cells']==6
    for key,value in [('chest_hybrid_study',False),('causal_scene_position_policy','original'),('causal_scene_memory',False),
                      ('object_state_memory_study',False),('object_state_text_control',None),('cut_scenario','envelope_revisit'),
                      ('constructor_mode','strict_checkpoint_no_parameter_init')]:
        bad=SimpleNamespace(**vars(good));setattr(bad,key,value)
        with pytest.raises(ValueError):validate_object_state_screen(bad,ROOT)


@pytest.mark.parametrize('seed',[20260925,20260926])
def test_exact_hybrid_CLI_preflight_without_GPU_or_output(tmp_path,seed):
    output=tmp_path/'not_created'
    args=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),'--assets',str(tmp_path/'unused'),
        '--output',str(output),'--object-protocol-only','--cut-scenario','chest_revisit','--seed',str(seed),
        '--native-local-frames','32','--cfg1-positive-cache-only','--fixed-adaln-warps','16','--fixed-adaln-stages','1',
        '--object-state-memory-study','--causal-scene-memory','--causal-scene-position-policy','recent_virtual',
        '--object-state-text-control','past_settled_restatement','--chest-hybrid-study']
    result=subprocess.run(args,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    d=json.loads(result.stdout);assert d['blocks']==16 and d['scene_cuts']==[2,6,12]
    assert d['protocol']['hybrid_registration'] is not None and not d['protocol']['Dense_only']
    assert not output.exists()
