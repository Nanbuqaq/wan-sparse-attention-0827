from copy import deepcopy

import pytest

from scripts.build_native_hybrid_restore_figure import rows_from_report


def report():
    return dict(status='pass',full_original_KV_hash_gates=True,recorded_kernel_recipes_applied=True,
        repeats=30,warmup_per_mode=5,
        recipe_selected_with_offline_witness=False,raw_KV_bytes=100,log_tensor_bytes=2,
        hybrid_checkpoint_states={'checkpoint':dict(total_retained_state_tensor_bytes=30,committed_checkpoint_chunks=1)},
        medians_s=dict(raw_pageable=1,raw_bounded_pinned=2,clean_log_replay=4,checkpoint=3),
        p95_s=dict(raw_pageable=1.1,raw_bounded_pinned=2.1,clean_log_replay=4.1,checkpoint=3.1))


def test_hybrid_rows_keep_common_raw_bytes_and_separate_timings():
    rows=rows_from_report(report())
    assert [r['state_tensor_bytes'] for r in rows]==[2,30,100,100]


@pytest.mark.parametrize('field',['full_original_KV_hash_gates','recorded_kernel_recipes_applied'])
def test_rejects_missing_exact_recipe_evidence(field):
    data=deepcopy(report());data[field]=False
    with pytest.raises(ValueError):rows_from_report(data)
