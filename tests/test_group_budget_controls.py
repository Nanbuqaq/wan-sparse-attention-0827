import pytest
from scripts.build_group_budget_controls import build


def test_new_controls_do_not_repeat_completed_original_seed_cases():
    suites, expected = build('a'*40)
    assert len(expected) == 14
    for suite in suites.values():
        old = [c for c in suite['cases'] if c['seed'] == 20260904]
        new = [c for c in suite['cases'] if c['seed'] == 20260908]
        assert len(old) == 2 and len(new) == 5
        assert all(c['history_density'] == .5 for c in old)
        assert sum(c['only_method'] == 'rag_dense' for c in new) == 1
        assert all(c['backend'] == 'resident_grouped_fa2' for c in suite['cases'])
        assert suite['formal_prompts_used'] is False
        assert suite['independent_timing_repeats'] is False


def test_reject_original_seed_as_replication():
    with pytest.raises(ValueError):
        build('a'*40, new_seed=20260904)
