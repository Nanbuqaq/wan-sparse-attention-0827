from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.calibrate_system_cost_model import calibrate


def _benchmark(path: Path, *, scale: float) -> None:
    results = {}
    h2d_per_byte = 2e-9
    launch = 3e-5
    pack_per_byte = 1e-9
    pack_per_run = 2e-5
    pack_fixed = 4e-5
    for index, (mode, copied, copies, pack_bytes, pack_runs) in enumerate(
        (
            ("direct_multirun", 1000, 8, 0, 0),
            ("packed_separate", 1200, 2, 1200, 4),
            ("packed_fused", 1200, 1, 1200, 4),
            ("direct_multirun", 4000, 16, 0, 0),
            ("packed_separate", 4800, 2, 4800, 8),
            ("packed_fused", 4800, 1, 4800, 8),
        )
    ):
        h2d = copied * h2d_per_byte + copies * launch
        pack = 0.0
        if pack_bytes or pack_runs:
            pack = pack_fixed + pack_bytes * pack_per_byte + pack_runs * pack_per_run
        results[f"case{index}"] = {
            "transfer_mode": mode,
            "transferred_bytes": copied * scale,
            "h2d_copy_count": copies,
            "h2d_s_median": h2d * scale,
            "cpu_gather_s_median": pack * scale,
            "pack_run_count": pack_runs,
            "pack_bytes": pack_bytes * scale,
        }
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "gpu": "test-gpu",
                "compute_capability": [9, 0],
                "torch": "test",
                "cuda": "test",
                "results": results,
            }
        ),
        encoding="utf-8",
    )


def test_cost_calibration_enforces_heldout_mape_gate(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    holdout = tmp_path / "holdout.json"
    _benchmark(calibration, scale=1.0)
    _benchmark(holdout, scale=1.0)
    payload = calibrate(
        [calibration],
        [holdout],
        profile_id="test-v1",
        model_version="nnls-v1",
        hbm_bytes_per_second=1e12,
    )
    assert payload["status"] == "pass"
    assert payload["cost_aware_admission_allowed"] is True
    assert payload["heldout_mape"] == pytest.approx(0.0, abs=1e-8)
    assert len(payload["profile"]["source_artifact_sha256"]) == 64


def test_cost_calibration_rejects_mixed_hardware(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    holdout = tmp_path / "holdout.json"
    _benchmark(calibration, scale=1.0)
    _benchmark(holdout, scale=1.0)
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    payload["gpu"] = "different-gpu"
    holdout.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="one hardware identity"):
        calibrate(
            [calibration],
            [holdout],
            profile_id="test-v1",
            model_version="nnls-v1",
            hbm_bytes_per_second=1e12,
        )
