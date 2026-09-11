import pytest
from scripts.prepare_geometry_wave_inputs import reference_payload


def test_reference_has_runner_identity_fields_and_preserves_external_hashes():
    hashes=['a'*64,'b'*64,'c'*64]
    result=reference_payload(':'.join(hashes))
    assert result['status']=='pass' and result['seed']==20260913
    assert result['latent_shape']==[1,128,48,44,80]
    assert len(result['prompts_per_block'])==16
    assert all(isinstance(x,str) and x for x in result['prompts_per_block'])
    assert [result['noise_sha256'],result['latent_sha256'],result['pixels']['raw_RGB_sha256']]==hashes


def test_incomplete_or_nonhex_reference_is_rejected():
    for value in ('', 'a'*64, ':'.join(['z'*64]*3)):
        with pytest.raises(ValueError):reference_payload(value)
