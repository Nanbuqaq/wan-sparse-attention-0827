from scripts.build_hierarchical_video_pairs import build


def test_hierarchical_stage_has_eight_matched_cases_not_different_gpu_controls():
    suites, expected = build('a'*40)
    assert len(expected['cases']) == 8
    assert len({r['case_key_sha256'] for r in expected['cases']}) == 8
    for suite in suites.values():
        left, right = suite['cases']
        assert (left['prompt'], left['seed'], left['latent_frames']) == (right['prompt'], right['seed'], right['latent_frames'])
        assert left['longlive_system']['gpu_union_cache_budget_mib'] == right['longlive_system']['gpu_union_cache_budget_mib'] == 4096
        assert {left['longlive_system']['gpu_union_cache'], right['longlive_system']['gpu_union_cache']} == {'per_chunk', 'hierarchical'}
        assert not suite['formal_prompts_used']


def test_raw_rgb_diagnostic_is_subset_and_distinct_case_identity():
    ordinary, _ = build('a'*40)
    diagnostic, expected = build('a'*40, raw_video_capture=True, lane_filter=(0, 1))
    assert len(expected['cases']) == 4
    assert set(diagnostic) == {0, 1}
    assert diagnostic[0]['cases'][0]['raw_video_capture']
    assert diagnostic[0]['cases'][0]['longlive_system']['profile_mode'] == 'trace'
    assert ordinary[0]['cases'][0]['longlive_system']['profile_mode'] == 'off'


def test_batched_backend_pair_preserves_all_route_and_cache_parameters():
    suites, expected = build('a'*40, pair_type='batched_backend')
    assert len(expected['cases']) == 8
    for suite in suites.values():
        left, right = suite['cases']
        assert {left['backend'], right['backend']} == {'grouped_fa2', 'batched_fa2'}
        for case in (left, right):
            assert case['longlive_system']['gpu_union_cache'] == 'per_chunk'
            assert case['longlive_system']['raw_cache_budget_mib'] == 0
