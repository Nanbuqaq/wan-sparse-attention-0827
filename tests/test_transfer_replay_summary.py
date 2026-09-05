from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_transfer_replay import summarize


def _benchmark(path: Path, *, direct_s: float, packed_s: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "gpu": "NVIDIA H200",
                "compute_capability": [9, 0],
                "capture": str(path),
                "source_preparation_s": 1.0,
                "results": {
                    "exact_compact__direct_multirun": {
                        "layout": "exact_compact",
                        "transfer_mode": "direct_multirun",
                        "staging_mode": None,
                        "gather_plus_h2d_s_median": direct_s,
                        "cpu_gather_s_median": 0.0,
                        "h2d_s_median": direct_s,
                        "transferred_bytes": 100,
                        "payload_bytes": 100,
                        "source_run_count": 4,
                        "h2d_copy_count": 8,
                    },
                    "exact_compact__persistent_separate": {
                        "layout": "exact_compact",
                        "transfer_mode": "packed_separate",
                        "staging_mode": "persistent_separate",
                        "gather_plus_h2d_s_median": packed_s,
                        "cpu_gather_s_median": packed_s - 0.1,
                        "h2d_s_median": 0.1,
                        "transferred_bytes": 100,
                        "payload_bytes": 100,
                        "source_run_count": 4,
                        "h2d_copy_count": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_transfer_summary_does_not_promote_microbenchmark(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _benchmark(first, direct_s=0.2, packed_s=0.3)
    _benchmark(second, direct_s=0.4, packed_s=0.35)
    payload = summarize([first, second])
    assert payload["gpu"] == "NVIDIA H200"
    assert payload["fastest_case_counts"]["exact_compact__direct_multirun"] == 1
    assert payload["promotion"]["pure_system_layout_promoted"] is False
