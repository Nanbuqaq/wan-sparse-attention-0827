import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from adapters.longlive_sparse.object_state_protocol import validate_object_state_screen

ROOT=Path(__file__).resolve().parents[1]


def test_layer_probe_is_isolated_from_lease_and_old_teacher():
    args=SimpleNamespace(cut_scenario='chest_revisit',seed=20260925,object_state_memory_study=True,
        object_state_text_control='past_settled_restatement',causal_scene_memory=True,
        causal_scene_position_policy='recent_virtual',chest_hybrid_study=True,chest_layer_role_probe=True)
    assert validate_object_state_screen(args,ROOT)['layer_role_probe_registration']['layer_count']==30
    for key,value in [('chest_source_pin_lease',True),('capture_attention_teacher',True),('chest_hybrid_study',False)]:
        changed=SimpleNamespace(**vars(args));setattr(changed,key,value)
        with pytest.raises(ValueError):validate_object_state_screen(changed,ROOT)


@pytest.mark.parametrize('seed',[20260925,20260926])
def test_exact_layer_probe_CLI_preflight_uses_reference_but_loads_no_model(tmp_path,seed):
    output=tmp_path/'not_created'
    args=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),'--assets',str(tmp_path/'unused'),
        '--output',str(output),'--object-protocol-only','--equivalence-reference',str(tmp_path/'not_read'),
        '--cut-scenario','chest_revisit','--seed',str(seed),'--native-local-frames','32','--cfg1-positive-cache-only',
        '--fixed-adaln-warps','16','--fixed-adaln-stages','1','--object-state-memory-study','--causal-scene-memory',
        '--causal-scene-position-policy','recent_virtual','--object-state-text-control','past_settled_restatement',
        '--chest-hybrid-study','--chest-layer-role-probe']
    result=subprocess.run(args,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['protocol']['layer_role_probe_registration'] is not None
    assert not output.exists()
