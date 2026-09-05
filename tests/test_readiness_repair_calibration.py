from scripts.build_readiness_repair_calibration import build


def test_repair_matrix_is_isolated_bounded_and_has_no_top_p_lane():
    suites,expected=build('a'*40)
    assert len(suites)==4 and len(expected['cases'])==8
    assert 'legacy_final_top_p095' not in suites
    for suite in suites.values():
        assert [case['prompt_id'] for case in suite['cases']]==['calibration_motion','calibration_state']
        for case in suite['cases']:
            system=case['longlive_system']
            assert system['archive_offload']=='pooled_pageable'
            assert system['host_pinned_budget_mib']==128
            assert system['cpu_pack_policy']=='archive_runs'
            assert system['gpu_union_cache']=='per_chunk'
