import torch
import pytest
from adapters.longlive_sparse.native_resident_history import summarize_frame,NativeResidentConfig
from adapters.longlive_sparse.native_summary_vectorized import summarize_frame_vectorized


@pytest.mark.parametrize('tokens',[48,64,128,880])
def test_vectorization_preserves_block_order_and_real_tail_counts(tokens):
    values=torch.arange(tokens*3*4).reshape(tokens,3,4).bfloat16()
    scalar=summarize_frame(values,values+10)
    vector=summarize_frame_vectorized(values,values+10)
    assert all(torch.equal(a,b) for a,b in zip(scalar,vector))


def test_backend_is_explicit_and_does_not_change_default():
    assert NativeResidentConfig().summary_backend=='scalar'
    with pytest.raises(ValueError):NativeResidentConfig(summary_backend='unknown')
