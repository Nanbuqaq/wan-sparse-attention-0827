import json
import pytest
import torch
from adapters.longlive_sparse.wave2_temporal_budget import Wave2TemporalBudget


def test_compact_age_record_rejects_current_or_future_optional_history():
    c=Wave2TemporalBudget.__new__(Wave2TemporalBudget)
    c.selector='query_sum_batch4';c.frame_tokens=128;c.age_records=[];c.age_bytes=0;c.age_host_s=0.
    row=dict(call=20,layer=14,current_frame=24,denoise_index=3,route_reused=True)
    c.record_age(row,[(1,('native',23,5,0.))],[[128]],False)
    record=json.loads(c.age_records[0]);assert record['optional_frame_ids']==[23] and record['route_reused']
    with pytest.raises(RuntimeError,match='uncommitted'):
        c.record_age(row,[(1,('native',24,5,0.))],[[128]],False)


def test_reused_route_statistics_do_not_claim_old_coverage_for_current_query():
    c=Wave2TemporalBudget.__new__(Wave2TemporalBudget)
    c.defer_stats=True;c.stats_D2H_bytes=0;c.stats_queue_bytes=16;c.stats_flush_host_s=0.
    row=dict(protected_union_tokens=2,route_reused=True)
    c.stats_queue=[(row,torch.tensor([1.,.3,.5,3.]),None,None,1,4,8)]
    c.flush_statistics()
    assert row['logical_pairs']==12 and row['selected_optional_tokens']==1
    assert row['coverage_min_per_head'] is None and row['coverage_mean_per_head'] is None
