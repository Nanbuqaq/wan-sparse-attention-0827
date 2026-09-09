import numpy as np
import pytest

from scripts.measure_key_position_color_proxy import normalize_occupancy


def test_occupancy_is_a_ratio_not_a_clipped_quality_score():
    ref,values=normalize_occupancy([.1,.2,.1],[0,.1,.3])
    assert ref==.1
    assert np.allclose(values,[0,1,3])


def test_no_red_source_is_not_silently_assigned_a_score():
    with pytest.raises(ValueError):normalize_occupancy([0,0],[.3])
