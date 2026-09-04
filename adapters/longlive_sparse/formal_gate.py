"""Validation gate for the new system holdout prompt manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FORMAL_SEEDS = [20260908, 20260909]
EXPECTED_LONG_SEEDS = [20260910]
EXPECTED_NON_STATE = {
    "identity_mars_astronaut",
    "human_glassblower",
    "fast_fox_snow",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_system_holdouts(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "frozen_before_sparse_system_video":
        errors.append("system holdouts are not frozen before sparse video")
    if payload.get("sparse_results_used") is not False:
        errors.append("system holdout freeze must not consume sparse results")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 4:
        errors.append("system holdout manifest must contain exactly four prompts")
        prompts = []
    ids = {str(item.get("prompt_id")) for item in prompts if isinstance(item, dict)}
    missing = sorted(EXPECTED_NON_STATE - ids)
    if missing:
        errors.append(f"system holdouts missing fixed prompts: {missing}")
    state_prompts = [
        item for item in prompts if isinstance(item, dict) and item.get("category") == "irreversible_state"
    ]
    if len(state_prompts) != 1:
        errors.append("system holdouts require exactly one irreversible-state prompt")
    if payload.get("formal_477_seeds") != EXPECTED_FORMAL_SEEDS:
        errors.append("formal 477 seeds do not match the frozen protocol")
    if payload.get("formal_957_seeds") != EXPECTED_LONG_SEEDS:
        errors.append("formal 957 seeds do not match the frozen protocol")
    source = payload.get("selection_source")
    if not isinstance(source, dict) or len(str(source.get("sha256", ""))) != 64:
        errors.append("selection_source must include a SHA-256 digest")
    dense_audit = payload.get("dense_terminal_audit")
    if not isinstance(dense_audit, dict) or len(str(dense_audit.get("sha256", ""))) != 64:
        errors.append("dense_terminal_audit must include a SHA-256 digest")
    if "state_melting_candle" in ids:
        errors.append("state_melting_candle is stress-only and cannot be a formal holdout")
    return errors


def load_frozen_system_holdouts(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            "new sparse video is blocked until system_holdout_prompts.json exists"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    errors = validate_system_holdouts(payload)
    if errors:
        raise ValueError("invalid system holdout manifest: " + "; ".join(errors))
    return payload
