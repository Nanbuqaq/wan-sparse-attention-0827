from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_query_policy_capture import summarize


def _artifact(path: Path, *, legacy_error: float, top_p_error: float) -> None:
    def record(error: float, density: float) -> dict:
        return {
            "route": {
                "history_pair_density": density,
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
                "mode": "query_policy",
                "capture_metadata": {"layer": 0, "current_start": 10},
                "records": {
                    "legacy_exact_union": record(legacy_error, 0.25),
                    "top_p_095": record(top_p_error, 0.18),
                },
            }
        ),
        encoding="utf-8",
    )


def test_query_policy_summary_is_quality_first_and_not_final_freeze(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _artifact(first, legacy_error=0.2, top_p_error=0.3)
    _artifact(second, legacy_error=0.25, top_p_error=0.35)
    payload = summarize([first, second])
    assert payload["preliminary_capture_winner"] == "legacy_exact_union"
    assert payload["physical_transfer_invariant_all_cases"] is True
    assert payload["final_policy_frozen"] is False
