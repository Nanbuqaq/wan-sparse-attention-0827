import json

from scripts.analyze_route_timeline import aggregate, summarize_case


def test_route_timeline_analysis_separates_pack_hash_attention(tmp_path):
    case = tmp_path / "case0"
    case.mkdir()
    (case / "summary.json").write_text(json.dumps({
        "status": "pass",
        "native_DiT_s": 2.0,
        "video_pipeline": {"complete_s": 3.0},
        "pixels": {"first_packet_s": 0.5, "raw_RGB_sha256": "r" * 64},
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
    }))
    row = summarize_case(case / "summary.json")
    assert row["device_interval_s"]["pack_attention_reshape.pack"] == 0.1
    assert row["device_interval_s"]["pack_attention_reshape.payload_hash"] == 0.02
    assert row["device_interval_s"]["pack_attention_reshape.attention"] == 0.3
    assert row["pair_ratio"] == 0.5
    assert aggregate([row])[0]["attention_device_s_median"] == 0.3
