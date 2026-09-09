import pytest

from scripts.review_key_position_study import selected_source_pixel_interval


def test_wrong_source_panel_is_not_the_related_red_state_panel():
    assert selected_source_pixel_interval(40)==(157,189)
    assert selected_source_pixel_interval(56)==(221,253)


def test_initial_or_empty_source_rejected():
    with pytest.raises(ValueError):selected_source_pixel_interval(0)
    with pytest.raises(ValueError):selected_source_pixel_interval(40,0)
