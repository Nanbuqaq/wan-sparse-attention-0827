from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_system_profile import summarize


def test_system_profile_summary_requires_same_route_and_keeps_service_boundary(tmp_path: Path) -> None:
    expected = []
    states = []
    for config_id, total, service in (
        ("legacy", 100.0, 50.0),
        ("exact_per_chunk_roped", 90.0, 40.0),
    ):
        key = config_id * 8
        expected.append(
            {
                "case_key_sha256": key,
                "profile_config_id": config_id,
                "latent_frames": 39,
            }
        )
        states.append(
            {
                "case_key_sha256": key,
                "status": "pass",
                "route_plan_sha256": "r",
                "route_plan_sha256s": ["a", "b"],
                "end_to_end_s": total,
                "routing_s": service / 2,
                "cpu_gather_s": service / 2,
                "h2d_s": 0.0,
                "attention_s": 5.0,
                "rope_s": 1.0,
                "transferred_bytes": 100,
                "candidate_transfer_bytes": 400,
                "peak_allocated_gb": 10.0,
            }
        )
    expected_path = tmp_path / "expected.json"
    states_path = tmp_path / "states.json"
    expected_path.write_text(json.dumps({"cases": expected}), encoding="utf-8")
    states_path.write_text(json.dumps({"cases": states}), encoding="utf-8")
    payload = summarize(states_path, expected_path)
    assert payload["status"] == "pass"
    assert payload["same_route_plan_all_system_comparisons"] is True
    assert payload["comparisons"][0]["route_gather_h2d_service_speedup"] == 1.25
    assert payload["promotion"]["pure_system_config"] is None
    assert "not measured exposed wait" in payload["evidence_boundary"]
