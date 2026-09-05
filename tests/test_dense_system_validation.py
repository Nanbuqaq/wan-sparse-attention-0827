from scripts.build_dense_system_validation import build
from adapters.longlive_sparse.case_identity import validate_case_identity


def test_development_system_comparison_is_dense_only_and_has_unique_identities():
    suites, expected = build('a'*40)
    assert len({case['case_key_sha256'] for case in expected['cases']}) == 4
    for suite, case in zip(suites, expected['cases']):
        assert suite['history_density'] == 1.0
        assert suite['methods'] == ['rag_dense']
        assert suite['formal_prompts_used'] is False
        assert not validate_case_identity(case)
