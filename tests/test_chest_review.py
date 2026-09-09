import pytest

from scripts.review_chest_causal_memory import source_pixel_interval


@pytest.mark.parametrize('frames,expected',[(list(range(40,48)),(157,189)),(list(range(8)),(0,29)),(list(range(88,96)),(349,381))])
def test_source_interval_uses_actual_causal_pixel_mapping(frames,expected):
    assert source_pixel_interval(frames)==expected


@pytest.mark.parametrize('frames',[[],[8,10]])
def test_reject_noncontiguous_source_interval(frames):
    with pytest.raises(ValueError):source_pixel_interval(frames)
