from scripts.build_group_relation_video_suite import build


def test_group_screen_preserves_controls_and_explicit_budget_boundary():
    suites, expected = build('a'*40, 39)
    assert len(expected) == 10
    for suite in suites.values():
        cases = suite['cases']
        assert len(cases) == 5 and not suite['formal_prompts_used']
        assert {c['longlive_system']['gpu_union_cache_budget_mib'] for c in cases} == {4096}
        conditional = [c for c in cases if c['method_params'].get('relation_admission') == 'per_group']
        assert len(conditional) == 2
        assert conditional[0]['method_params'] == conditional[1]['method_params']
        assert {c['backend'] for c in conditional} == {'resident_grouped_fa2', 'grouped_fa2'}
        assert all(c['history_density'] == .25 for c in cases if c['only_method'] != 'rag_dense')
