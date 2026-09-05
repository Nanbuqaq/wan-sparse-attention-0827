from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_static_utility_capture import summarize


def _artifact(path: Path, errors: dict[str, float]) -> None:
    records = {
        "legacy_final_reference": {
            "online_route_s": 0.0,
            "route": {
                "history_pair_density": 0.25,
                "history_transfer_density": 0.25,
            },
            "history_only_output_error": {
                "relative_l2": 0.3,
                "one_minus_cosine": 0.15,
            },
            "transfer_execution": {"copied_bytes": 100},
        }
    }
    for name, error in errors.items():
        records[f"{name}__static_block"] = {
            "online_route_s": 0.1,
            "route": {
                "history_pair_density": 0.25,
                "history_transfer_density": 0.25,
            },
            "history_only_output_error": {
                "relative_l2": error,
                "one_minus_cosine": error / 2,
            },
            "transfer_execution": {"copied_bytes": 100},
        }
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "mode": "utility",
                "marginal_candidates_evaluated": False,
                "capture_metadata": {"layer": 0, "current_start": 1},
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def test_static_utility_summary_retains_two_without_final_freeze(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _artifact(first, {"sum_value": 0.2, "peak_value": 0.3, "count_uniform": 0.4})
    _artifact(second, {"sum_value": 0.25, "peak_value": 0.35, "count_uniform": 0.45})
    payload = summarize([first, second])
    assert payload["retained_for_motion_state_calibration"] == [
        "sum_value__static_block",
        "peak_value__static_block",
    ]
    assert payload["final_utility_frozen"] is False
    assert payload["marginal_cost_candidates_status"].startswith("stopped")
    assert payload["aggregate"]["sum_value__static_block"][
        "not_worse_than_legacy_all_captures"
    ] is True
