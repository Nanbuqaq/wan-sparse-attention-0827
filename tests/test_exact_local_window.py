import pytest
from adapters.longlive_sparse.config import SparseHistoryConfig


def test_local_reallocation_is_a_method_parameter_not_a_transfer_layout():
    base=SparseHistoryConfig(method='transfer_vaware_hybrid_history')
    modified=SparseHistoryConfig(method='transfer_vaware_hybrid_history',method_params={'exact_local_window_frames':8})
    assert base.as_dict()!=modified.as_dict() and modified.method_params['exact_local_window_frames']==8


@pytest.mark.parametrize('value',[0,-1,3.5,True])
def test_invalid_local_window_rejected(value):
    with pytest.raises(ValueError):SparseHistoryConfig(method_params={'exact_local_window_frames':value})
