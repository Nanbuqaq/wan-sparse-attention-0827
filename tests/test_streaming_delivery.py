import pytest

from scripts.summarize_streaming_delivery import delivery_metrics


def test_delivery_not_service_and_late_window():
    records=[dict(start_latent=i*3,sink_completed_s=i*2.,pixel_frames=12) for i in range(15)]
    result=delivery_metrics(records,start_latent=30)
    assert result['chunks']==5 and result['delivery_gap_p50_s']==2
    assert result['fraction_meeting_per_chunk_playback_budget']==0
    records[-1]['sink_completed_s']=0
    with pytest.raises(ValueError):delivery_metrics(records,start_latent=30)
