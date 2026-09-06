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
