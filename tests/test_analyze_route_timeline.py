import json

import pytest

from scripts.analyze_route_timeline import aggregate, summarize_case, summarize_delivery


def _summary_payload():
    return {
        "status": "pass",
        "native_DiT_s": 2.0,
        "video_pipeline": {
            "complete_s": 3.0,
            "producer_backpressure_s": 0.01,
            "records": [
                {
                    "submitted_s": 0.5,
                    "decode_started_s": 0.6,
                    "H2D_GPU_stream_span_ms": 2.0,
                    "D2H_GPU_stream_span_ms": 0.1,
                    "groups": [
                        {"decode_GPU_stream_span_ms": 100.0, "pixel_D2H_GPU_stream_span_ms": 1.0, "CPU_pixels_ready_s": 0.8, "pixel_D2H_bytes": 4},
                        {"decode_GPU_stream_span_ms": 120.0, "pixel_D2H_GPU_stream_span_ms": 1.0, "CPU_pixels_ready_s": 2.5, "pixel_D2H_bytes": 4},
                    ],
                }
            ],
        },
        "pixels": {"first_packet_muxed_s": 0.5, "raw_RGB_sha256": "r" * 64},
        "wave2": {
            "method": "w2_steady_sparse",
            "selector": "query_sum_batch4",
            "preparation": "geometry_cache",
            "route_audit_sha256": "a" * 64,
            "route_timeline_records": [{
                "state": "steady_sparse",
                "prepare_host_s": 0.1,
                "execution_submit_host_s": 0.2,
                "packed_QKV_bytes": 12,
                "pack_attention_reshape_interval_device_s": [0.1, 0.02, 0.3, 0.01],
            }],
            "rows": [{"logical_pairs": 50, "full_native_pairs": 100, "selected_mask_sha256": "b" * 64}],
        },
    }


def test_route_timeline_analysis_separates_pack_hash_attention(tmp_path):
    case = tmp_path / "case0"
    case.mkdir()
    (case / "summary.json").write_text(json.dumps(_summary_payload()))
    row = summarize_case(case / "summary.json")
    assert row["device_interval_s"]["pack_attention_reshape.pack"] == 0.1
    assert row["device_interval_s"]["pack_attention_reshape.payload_hash"] == 0.02
    assert row["device_interval_s"]["pack_attention_reshape.attention"] == 0.3
    assert row["pair_ratio"] == 0.5
    assert aggregate([row])[0]["attention_device_s_median"] == 0.3


def test_first_packet_uses_muxed_field(tmp_path):
    case = tmp_path / "case0"
    case.mkdir()
    (case / "summary.json").write_text(json.dumps(_summary_payload()))
    row = summarize_case(case / "summary.json")
    assert row["first_packet_s"] == 0.5
    assert aggregate([row])[0]["first_packet_median_s"] == 0.5


def test_delivery_summary_separates_service_queue_and_tail():
    delivery = summarize_delivery(_summary_payload()["video_pipeline"], 2.0)
    assert delivery["records"] == 1
    assert delivery["groups"] == 2
    assert delivery["decode_service_s"] == pytest.approx(0.22)
    assert delivery["pixel_D2H_service_s"] == pytest.approx(0.002)
    assert delivery["latent_H2D_service_s"] == pytest.approx(0.002)
    assert delivery["chunk_queue_wait_median_s"] == pytest.approx(0.1)
    assert delivery["decode_window_s"] == pytest.approx(1.9)
    assert delivery["delivery_tail_s"] == pytest.approx(1.0)
    # Only the second group is ready after generation end at 2.0s.
    assert delivery["post_generation_decode_service_s_approx"] == pytest.approx(0.12)
    assert delivery["producer_backpressure_s"] == 0.01

