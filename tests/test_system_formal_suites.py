from __future__ import annotations

import json
import subprocess
from pathlib import Path

from adapters.longlive_sparse.system_formal import validate_system_method_freeze
from scripts.build_system_formal_suites import build


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


def _freeze(path: Path) -> dict:
    configs = []
    for config_id, method, density in (
        ("rag_dense", "rag_dense", 1.0),
        ("legacy_final", "transfer_vaware_hybrid_history", 0.25),
        ("legacy_final_system", "transfer_vaware_hybrid_history", 0.25),
        ("best_causal_or_codesign", "transfer_vaware_hybrid_history", 0.25),
    ):
        configs.append(
            {
                "config_id": config_id,
                "method": method,
                "backend": "packed_fa2" if method == "rag_dense" else "grouped_fa2",
                "history_density": density,
                "online": True,
                "longlive_system": {},
            }
        )
    payload = {
        "artifact_id": "freeze",
        "status": "frozen_after_system_calibration",
        "formal_results_used": False,
        "calibration_audit": {"sha256": "c" * 64},
        "profile_audit": {"sha256": "d" * 64},
        "quality_gate": {"sha256": "e" * 64},
        "configs": configs,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_method_freeze_rejects_oracle_and_builds_32_16_cases(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    holdouts = tmp_path / "holdouts.json"
    freeze = tmp_path / "freeze.json"
    _holdouts(holdouts)
    payload = _freeze(freeze)
    assert validate_system_method_freeze(payload) == []
    suites_477, expected_477 = build(
        holdout_path=holdouts,
        method_freeze_path=freeze,
        method_params_path=root / "configs/formal/method_params.json",
        latent_frames=120,
        commit="a" * 40,
    )
    suites_957, expected_957 = build(
        holdout_path=holdouts,
        method_freeze_path=freeze,
        method_params_path=root / "configs/formal/method_params.json",
        latent_frames=240,
        commit="a" * 40,
    )
    assert len(suites_477) == len(suites_957) == 4
    assert len(expected_477["cases"]) == 32
    assert len(expected_957["cases"]) == 16
    payload["configs"][-1]["method"] = "tethermem_oracle_mask_teacher"
    assert any("oracle" in error for error in validate_system_method_freeze(payload))


def test_formal_batch_is_eight_real_lanes_and_requires_both_freezes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_system_formal_8gpu.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")
    assert "validate_system_holdout_prompts.py" in source
    assert "validate_system_method_freeze.py" in source
    assert "for lane in 0 1 2 3 4 5 6 7" in source
    assert "config_index=$((lane / 2))" in source
    assert "SYSTEM_FORMAL_LATENT_FRAMES" in source
