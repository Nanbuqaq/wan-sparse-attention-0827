import torch
import pytest
from scripts.probe_wave2_incremental_groups import replay


def test_identical_features_merge_and_eager_delayed_have_identical_final_directory():
    data={'arrivals':[{'frame':f,'key_features':torch.ones(32,128)} for f in (0,8)],
          'accesses':[{'frame':8,'ranges':[[0,32]]}]}
    result=replay(data,4)
    assert result['K_feature_groups']==1 and result['raw_tokens']==64
    a,b=result['maintenance']
    assert a['final_intervals']==b['final_intervals']==1
    assert a['index_integer_writes']==b['index_integer_writes']
    assert b['dirty_tokens_before_drain']==32


def test_future_access_is_rejected_even_if_future_assignment_is_known_to_replay():
    data={'arrivals':[{'frame':f,'key_features':torch.ones(32,128)} for f in (0,8)],
          'accesses':[{'frame':0,'ranges':[[0,1]]}]}
    with pytest.raises(ValueError,match='committed'):replay(data,4)
