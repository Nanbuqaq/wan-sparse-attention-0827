import pytest

from scripts.analyze_region_layout_footprint import footprint


def test_full_roi_preserves_all_payload_and_reports_padding():
    rows=footprint(range(7040))
    assert all(r['payload_tokens']==7040 for r in rows)
    assert all(r['computed_payload_bytes']==2595225600 for r in rows)
    assert next(r for r in rows if r['layout']=='frame_linear_Block64')['computed_padding_bytes']>0


def test_fixed_roi_never_loses_logical_tokens():
    rows=footprint([460,461,500,501,880+460])
    assert all(r['logical_tokens']==5 and r['payload_tokens']>=5 for r in rows)
    assert rows[0]['physical_runs']==3


def test_invalid_roi_is_rejected():
    with pytest.raises(ValueError):footprint([7040])
