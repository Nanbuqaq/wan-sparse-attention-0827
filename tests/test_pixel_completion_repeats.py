from copy import deepcopy

import pytest

from scripts.summarize_pixel_completion_repeats import summarize_pairs


def fixtures():
    return [dict(status='pass',rows=[dict(mode=mode,status='pass',actual_full_latent_and_decoded_RGB_exact=True,
        pixel_slots_not_reused_before_sink_completion=True,frames=509,complete_s=seconds+i,first_packet_s=5.,
        producer_backpressure_s=0.) for mode,seconds in [('inline',100.),('thread',85.)]]) for i in range(3)]


def test_all_pairs_and_first_packet_are_retained():
    d=summarize_pairs(fixtures())
    assert d['modes']['inline']['median_complete_s']==101 and d['modes']['thread']['median_complete_s']==86
    assert d['all_three_pairs_exceed_10percent'] and d['modes']['thread']['median_first_packet_s']==5


def test_failed_pair_is_not_dropped_to_improve_mean():
    rows=deepcopy(fixtures());rows[1]['status']='fail'
    with pytest.raises(ValueError,match='successful-only'):summarize_pairs(rows)
