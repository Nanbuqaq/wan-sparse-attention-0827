from scripts.build_system_formal_by_group import regroup


def test_formal_grouping_keeps_same_seed_three_configs_on_one_lane():
    suites = {}
    expected = {'cases': []}
    for name, method in [('rag_dense', 'rag_dense'), ('legacy_final', 'transfer_vaware_hybrid_history'),
                         ('legacy_final_system', 'transfer_vaware_hybrid_history')]:
        cases = [{'prompt_id': prompt, 'seed': seed, 'formal_config_id': name} for prompt in ('a', 'b') for seed in (7, 8)]
        suites[name] = {'methods': [method], 'method_params': {method: {}}, 'cases': cases}
        expected['cases'].extend({**c, 'method': method} for c in cases)
    grouped, manifest = regroup(suites, expected)
    assert len(grouped) == 4
    assert len(manifest['cases']) == 12
    for lane, suite in grouped.items():
        assert len(suite['cases']) == 3
        assert len({(c['prompt_id'], c['seed']) for c in suite['cases']}) == 1
        assert suite['methods'] == ['rag_dense', 'transfer_vaware_hybrid_history']
        assert suite['cases'][0]['only_method'] == 'rag_dense'
    assert grouped[0]['within_group_config_order'] != grouped[1]['within_group_config_order']
