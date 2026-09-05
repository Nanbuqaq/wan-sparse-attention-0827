from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.build_system_profile_suite import build


def _holdouts(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_id": "test",
                "status": "frozen_before_sparse_system_video",
                "sparse_results_used": False,
                "formal_477_seeds": [20260908, 20260909],
                "formal_957_seeds": [20260910],
                "selection_source": {"sha256": "a" * 64},
                "dense_terminal_audit": {"sha256": "b" * 64},
                "prompts": [
                    {"prompt_id": "identity_mars_astronaut", "category": "identity_scene"},
                    {"prompt_id": "human_glassblower", "category": "human_action"},
                    {"prompt_id": "fast_fox_snow", "category": "fast_motion"},
                    {"prompt_id": "state_blue_canvas_paint", "category": "irreversible_state"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_profile_suite_is_gated_isolated_and_twelve_cases(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    holdouts = tmp_path / "holdouts.json"
    _holdouts(holdouts)
    suite, expected = build(
        holdout_path=holdouts,
        prompt_path=root / "configs/system/profile_calibration_prompts.json",
        method_params_path=root / "configs/formal/method_params.json",
        commit="a" * 40,
    )
    assert suite["formal_prompts_used"] is False
    assert len(suite["cases"]) == len(expected["cases"]) == 12
    assert {case["latent_frames"] for case in suite["cases"]} == {39, 120, 240}
    assert {case["profile_config_id"] for case in suite["cases"]} == {
        "legacy",
        "exact_per_chunk_roped",
        "block64_per_chunk_roped",
        "exact_cross_chunk_raw",
    }
    assert all(case["seed"] == 20260904 for case in suite["cases"])


def test_system_profile_batch_is_four_lane_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_system_profile_4gpu.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")
    assert "validate_system_holdout_prompts.py" in source
    assert "LONGLIVE_CAPTURE_ROUTE_LAYERS=0,9,19,29" in source
    assert "--shard-axis case" in source
    assert "--shard-count 4" in source
