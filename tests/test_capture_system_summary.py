from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_capture_system import summarize


def _write(path: Path, *, layer: int, start: int) -> None:
    payload = {
        "status": "pass",
        "capture_metadata": {
            "layer": layer,
            "current_start": start,
            "history_shape": [1, 10, 2, 4],
        },
        "route": {
            "route_plan_sha256": "a" * 64,
            "history_pair_density": 0.25,
            "history_transfer_density": 0.25,
        },
        "transfer_layouts": {
            "exact_compact": {"physical_copy_bytes": 100, "padding_bytes": 0, "source_run_count": 10},
            "block64": {"physical_copy_bytes": 110, "padding_bytes": 10, "source_run_count": 8},
            "page256": {"physical_copy_bytes": 180, "padding_bytes": 80, "source_run_count": 5},
            "frame1560": {"physical_copy_bytes": 400, "padding_bytes": 300, "source_run_count": 2},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_requires_complete_matrix_and_reports_tradeoff(tmp_path: Path) -> None:
    paths = []
    for layer in (0, 9):
        path = tmp_path / f"layer{layer}.json"
        _write(path, layer=layer, start=100)
        paths.append(path)
    payload = summarize(paths, expected_cases=2)
    assert payload["status"] == "pass"
    assert payload["aggregate"]["block64"]["byte_multiplier_vs_exact_mean"] == 1.1
    assert payload["aggregate"]["block64"]["cases_with_fewer_runs"] == 2
    incomplete = summarize(paths, expected_cases=3)
    assert incomplete["status"] == "incomplete"
