from scripts.build_system_formal_by_group import regroup


def test_frozen_modes_smoke_never_uses_holdout_frames():
    import json
    from pathlib import Path
    from scripts.build_system_formal_by_group import runtime_smoke
    from scripts.build_system_formal_suites import build
    root = Path(__file__).resolve().parents[1]
    suites, _ = build(holdout_path=root/'configs/formal/system_holdout_prompts.json',
        method_freeze_path=root/'configs/formal/system_method_freeze.json',
        method_params_path=root/'configs/formal/method_params.json', latent_frames=120, commit='a'*40)
    calibration = json.loads((root/'configs/system/profile_calibration_prompts.json').read_text())
    smoke, expected = runtime_smoke(suites, calibration, 'a'*40)
    lanes, manifest = regroup(smoke, expected)
    assert len(lanes) == 2 and len(manifest['cases']) == 6
    assert len({c['id'] for c in manifest['cases']}) == 6
    for lane in lanes.values():
        assert lane['formal_prompts_used'] is False
        assert all(c['prompt_id'].startswith('calibration_') and c['latent_frames'] == 39 for c in lane['cases'])
        for c in lane['cases']:
            assert c['longlive_system'] == suites[c['formal_config_id']]['cases'][0]['longlive_system']


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
