import pytest

from scripts.audit_native_pixel_completion import validate_groups


def test_pixel_slot_reuse_requires_prior_sink_completion():
    groups=[dict(start_pixel=0,pixel_frames=1,pixel_buffer_slot=0,pixel_buffer_acquired_s=0.,CPU_pixels_ready_s=1.,sink_finished_s=3.),
            dict(start_pixel=1,pixel_frames=4,pixel_buffer_slot=0,pixel_buffer_acquired_s=3.1,CPU_pixels_ready_s=4.,sink_finished_s=5.)]
    stream=dict(pixel_buffer_slots=1,completed_pixel_frames=5,encoded_pixels=5,records=[dict(groups=groups,finished_s=5.)])
    assert len(validate_groups(stream))==2
    groups[1]['pixel_buffer_acquired_s']=2.
    with pytest.raises(AssertionError):validate_groups(stream)


def test_different_slots_can_overlap_but_sink_order_is_complete():
    groups=[dict(start_pixel=0,pixel_frames=1,pixel_buffer_slot=0,pixel_buffer_acquired_s=0.,CPU_pixels_ready_s=1.,sink_finished_s=3.),
            dict(start_pixel=1,pixel_frames=4,pixel_buffer_slot=1,pixel_buffer_acquired_s=1.1,CPU_pixels_ready_s=2.,sink_finished_s=5.)]
    stream=dict(pixel_buffer_slots=2,completed_pixel_frames=5,encoded_pixels=5,records=[dict(groups=groups,finished_s=5.)])
    assert len(validate_groups(stream))==2
