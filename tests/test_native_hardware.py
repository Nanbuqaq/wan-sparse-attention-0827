import pytest
from adapters.longlive_sparse.native_hardware import validate_hardware_names


def test_h200_stays_default_and_h800_needs_explicit_opt_in():
    assert validate_hardware_names(['NVIDIA H200']*4,expected_count=4)['actual_model']=='NVIDIA H200'
    with pytest.raises(ValueError):validate_hardware_names(['NVIDIA H800']*4,expected_count=4)
    result=validate_hardware_names(['NVIDIA H800']*4,expected_count=4,allow_h800=True)
    assert result['actual_model']=='NVIDIA H800' and result['H800_is_not_labeled_H200']


def test_unsupported_mixed_or_incomplete_allocations_fail_before_model_work():
    for names in (['NVIDIA A100']*4,['NVIDIA H800']*2,['NVIDIA H800']*2+['NVIDIA H200']*2,[]):
        with pytest.raises(ValueError):validate_hardware_names(names,expected_count=4,allow_h800=True)
