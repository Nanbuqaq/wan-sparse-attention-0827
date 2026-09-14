import pytest
import torch
from adapters.longlive_sparse.native_inplace_gelu import InplaceNativeGELU,NativeInplaceGelu


def test_native_operator_output_reuse_is_bitwise_equal_and_keeps_residual_input():
    torch.manual_seed(2)
    model=torch.nn.Module();model.blocks=torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].ffn=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.GELU(approximate='tanh'),torch.nn.Linear(32,8)).eval()
    model.eval();x=torch.randn(2,4,8);old=x.clone()
    with torch.inference_mode():
        expected=model.blocks[0].ffn(x)
        patch=NativeInplaceGelu(model,expected_layers=1)
        actual=model.blocks[0].ffn(x)
    assert torch.equal(expected,actual) and torch.equal(x,old)
    assert patch.audit()['calls']==1


def test_training_and_unqualified_observer_are_rejected():
    module=InplaceNativeGELU('tanh').eval()
    with pytest.raises(RuntimeError):module(torch.ones(4))
    model=torch.nn.Module();model.blocks=torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].ffn=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.GELU(),torch.nn.Linear(32,8));model.eval()
    model.blocks[0].ffn[0].register_forward_hook(lambda *x:None)
    with pytest.raises(ValueError):NativeInplaceGelu(model,expected_layers=1)


def test_common_optimization_reaches_native_and_candidates_with_first_case_guard():
    from scripts.run_native_duration_wave import with_common_inplace_gelu
    cases=[dict(method='native',cmd=['python','runner']),dict(method='value16',cmd=['python','runner'])]
    updated=with_common_inplace_gelu(cases,'old-native.json')
    assert all('--native-inplace-gelu' in x['cmd'] for x in updated)
    assert updated[0]['guarded_native_equivalence'] is True
    assert '--equivalence-reference' not in updated[1]['cmd'] and cases[0]['cmd']==['python','runner']
