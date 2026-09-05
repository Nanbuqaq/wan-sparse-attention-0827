from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.build_system_routing_calibration_suites import build


def _holdouts(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_id": "holdouts",
                "status": "frozen_before_sparse_system_video",
                "sparse_results_used": False,
                "formal_477_seeds": [20260908, 20260909],
                "formal_957_seeds": [20260910],
                "selection_source": {"sha256": "a" * 64},
                "dense_terminal_audit": {"sha256": "b" * 64},
                "prompts": [
                    {"prompt_id": "identity_mars_astronaut", "category": "identity_scene", "prompt": "i"},
                    {"prompt_id": "human_glassblower", "category": "human_action", "prompt": "h"},
                    {"prompt_id": "fast_fox_snow", "category": "fast_motion", "prompt": "f"},
                    {"prompt_id": "state_blue_canvas_paint", "category": "irreversible_state", "prompt": "s"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_routing_calibration_is_ten_isolated_cases(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    holdouts = tmp_path / "holdouts.json"
    _holdouts(holdouts)
    suites, expected = build(
        holdout_path=holdouts,
        prompt_path=root / "configs/system/profile_calibration_prompts.json",
        method_params_path=root / "configs/formal/method_params.json",
        candidate_path=root / "configs/system/capture_screened_candidates.json",
        commit="a" * 40,
    )
    assert len(suites) == 5
    assert len(expected["cases"]) == 10
    assert len({row["case_key_sha256"] for row in expected["cases"]}) == 10
    assert all(case["latent_frames"] == 39 for suite in suites.values() for case in suite["cases"])
    assert all(case["seed"] == 20260904 for suite in suites.values() for case in suite["cases"])
    assert suites["system_utility_peak"]["method_params"]["system_utility_history"][
        "cost_strategy"
    ] == "static_block"


def test_routing_calibration_batch_uses_five_real_gpu_lanes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_system_routing_calibration_5gpu.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")
    assert "requires exactly five GPUs" in source
    assert "for lane in 0 1 2 3 4" in source
    assert "validate_system_holdout_prompts.py" in source
    assert "system_utility_peak" in source
    assert "legacy_final_top_p095" in source
