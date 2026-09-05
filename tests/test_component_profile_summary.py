from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_component_profiles import summarize


def _state(path: Path, *, attention: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "end_to_end_s": 100.0,
                "prompt_id": "test",
                "seed": 1,
                "latent_frames": 39,
                "routing_s": 15.0,
                "cpu_gather_s": 20.0,
                "h2d_s": 5.0,
                "attention_s": attention,
            }
        ),
        encoding="utf-8",
    )


def test_component_summary_separates_service_from_exposed_wait(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _state(first, attention=5.0)
    _state(second, attention=6.0)
    payload = summarize([first, second])
    assert payload["aggregate"]["cpu_route_gather"]["fraction_median"] == 0.35
    assert payload["kvout_video_gate_currently_justified"] is False
    assert payload["evidence_boundary"]["measured_exposed_wait_available"] is False
