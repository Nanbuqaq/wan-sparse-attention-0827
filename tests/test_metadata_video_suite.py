import pytest

from scripts.build_metadata_video_suite import build


@pytest.mark.parametrize('length', [39, 120, 240])
def test_metadata_ablation_matches_methods_and_actual_budgets(length):
    suites, expected = build('a' * 40, length)
    assert len(expected) == 8
    assert len({r['case_key_sha256'] for r in expected}) == 8
    for suite in suites.values():
        assert not suite['formal_prompts_used']
        for method in suite['methods']:
            cases = [c for c in suite['cases'] if c['only_method'] == method]
            assert len(cases) == 2
            controls = []
            for case in cases:
                config = dict(case['longlive_system'])
                config.pop('route_metadata_mode')
                controls.append(config)
            assert controls[0] == controls[1]
            assert cases[0]['prompt'] == cases[1]['prompt']
            assert cases[0]['seed'] == cases[1]['seed']
