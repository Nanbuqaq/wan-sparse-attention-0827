import pytest

from scripts.audit_native_pipeline_gate import reference_geometry


def test_small_gate_preserves_existing_expected_bytes():
    got=reference_geometry(dict(latent_shape=[1,64,48,16,32],pixel_frames=253))
    assert got['latent_bytes']==3145728
    assert got['pixel_D2H_bytes']==397934592
    assert got['chunk_starts']==list(range(0,64,8))


def test_full_native_gate_is_not_a_scaled_small_gate_claim():
    got=reference_geometry(dict(latent_shape=[1,128,48,44,80],pixel_frames=509))
    assert (got['height'],got['width'])==(704,1280)
    assert got['pixel_D2H_bytes']==509*704*1280*3*4
    assert got['latent_bytes']==128*48*44*80*2
    assert len(got['chunk_starts'])==16


@pytest.mark.parametrize('shape,pixels',[
    ([1,64,48,16,32],254),([1,63,48,16,32],249),
    ([2,64,48,16,32],253),([1,64,16,16,32],253),
    ([1,64,48,0,32],253),
])
def test_geometry_mismatch_is_not_silently_accepted(shape,pixels):
    with pytest.raises(ValueError):
        reference_geometry(dict(latent_shape=shape,pixel_frames=pixels))
