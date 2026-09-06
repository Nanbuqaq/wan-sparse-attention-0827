import pytest
from scripts.run_loaded_method_suite import eligible_case_method, require_matched_reference


def test_case_method_filter_preserves_default_and_is_explicit():
    assert eligible_case_method({}, 'rag_dense')
    assert eligible_case_method({'only_method': 'rag_dense'}, 'rag_dense')
    assert not eligible_case_method({'only_method': 'rag_dense'}, 'tethermem_oracle_mask_teacher')


def test_oracle_blocks_unmatched_dense_and_fails_before_generation():
    case = {'prompt_id': 'state', 'seed': 3, 'latent_frames': 39, 'required_reference_video_sha256': 'a'*64}
    with pytest.raises(ValueError, match='bitwise-matched'):
        require_matched_reference(case, [])
    reference = {'method': 'rag_dense', 'prompt_id': 'state', 'seed': 3, 'latent_frames': 39,
                 'status': 'pass', 'video_sha256': 'a'*64}
    require_matched_reference(case, [reference])
    with pytest.raises(ValueError, match='bitwise-matched'):
        require_matched_reference(case, [{**reference, 'video_sha256': 'b'*64}])
