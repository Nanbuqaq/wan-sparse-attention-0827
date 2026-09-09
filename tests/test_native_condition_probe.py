from pathlib import Path

from scripts.probe_native_condition_retrieval import condition_probe_cases


def test_text_probe_uses_actual_closed_scene_conditions_and_explicit_counterfactuals():
    cases=condition_probe_cases(Path(__file__).resolve().parents[1])
    assert len(cases)==3
    for case in cases:
        assert [r['role'] for r in case['candidates']]==['initial','source','away']
        assert [q['expected_candidate'] for q in case['queries']]==['source','initial','away']
        assert case['candidates'][1]['end']==48
        assert all(q['text'].startswith('The scene transitions. ') for q in case['queries'])
    settled=next(c for c in cases if c['scenario']=='settled_bead_revisit')
    assert 'completely stopped' in settled['candidates'][1]['text']
