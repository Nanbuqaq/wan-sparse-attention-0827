from scripts.audit_case_states import expected_latent_frames
from adapters.longlive_sparse.case_identity import build_case_identity


def test_expected_shape_comes_from_canonical_identity_without_redundant_field():
    record = build_case_identity(commit='a'*40, method='rag_dense', prompt_id='test', prompt='test',
        seed=1, latent_frames=120, history_density=1., rope_policy='upstream_zero',
        refresh_policy='per_chunk', backend='grouped_fa2')
    assert 'latent_frames' not in record
    assert expected_latent_frames(record) == 120
