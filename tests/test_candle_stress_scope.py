import json
from pathlib import Path
import subprocess
import sys


def test_candle_stress_is_separate_from_formal_means_and_uses_new_seeds(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path/'control'
    subprocess.run([sys.executable, str(root/'scripts/build_system_candle_stress.py'), '--output-dir', str(out)], check=True)
    expected = json.loads((out/'expected.json').read_text())
    assert expected['exclude_from_formal_mean_pareto_selection'] is True
    assert len(expected['cases']) == len({c['id'] for c in expected['cases']}) == 6
    assert {c['seed'] for c in expected['cases']} == {20260911, 20260912}
    assert {c['prompt_id'] for c in expected['cases']} == {'state_melting_candle'}
    for lane in range(2):
        suite = json.loads((out/f'lane{lane}.json').read_text())
        assert suite['formal_prompts_used'] is False
        assert suite['exclude_from_formal_mean_pareto_selection'] is True
        assert len(suite['cases']) == 3
