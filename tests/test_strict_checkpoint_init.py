import pytest
import torch

from adapters.longlive_sparse.strict_checkpoint_init import StrictCheckpointParameterInit


def test_only_direct_parameters_are_skipped_not_tensors_or_views():
    parameter=torch.nn.Parameter(torch.full((2,3),7.));buffer=torch.zeros(2,3)
    original=torch.nn.init.normal_
    with StrictCheckpointParameterInit(enabled=True) as guard:
        assert torch.nn.init.normal_(parameter) is parameter
        assert torch.equal(parameter,torch.full((2,3),7.))
        torch.nn.init.ones_(buffer);assert torch.equal(buffer,torch.ones(2,3))
        torch.nn.init.zeros_(parameter.view(-1));assert torch.equal(parameter,torch.zeros(2,3))
    assert torch.nn.init.normal_ is original
    assert guard.record()['skipped_parameter_initializer_calls']==1


def test_complete_strict_checkpoint_restores_identical_parameters_and_output():
    torch.manual_seed(7);reference=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.LayerNorm(5),torch.nn.Linear(5,2))
    checkpoint={k:v.clone() for k,v in reference.state_dict().items()}
    with StrictCheckpointParameterInit(enabled=True) as guard:
        actual=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.LayerNorm(5),torch.nn.Linear(5,2))
        loaded=actual.load_state_dict(checkpoint,strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
    x=torch.arange(9).float().reshape(3,3)
    assert torch.equal(actual(x),reference(x))
    assert guard.record()['skipped_parameter_initializer_calls']>0


def test_missing_checkpoint_fails_and_restores_global_initializers():
    function=torch.nn.init.kaiming_uniform_
    with pytest.raises(RuntimeError):
        with StrictCheckpointParameterInit(enabled=True):
            torch.nn.Linear(3,2).load_state_dict({},strict=True)
    assert torch.nn.init.kaiming_uniform_ is function


def test_disabled_guard_preserves_reference_initialization():
    torch.manual_seed(2);a=torch.nn.Linear(3,2)
    torch.manual_seed(2)
    with StrictCheckpointParameterInit() as guard:b=torch.nn.Linear(3,2)
    assert torch.equal(a.weight,b.weight) and not guard.record()['enabled'] and not guard.rows
