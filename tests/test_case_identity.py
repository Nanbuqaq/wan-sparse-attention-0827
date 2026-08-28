from __future__ import annotations

import pytest

from adapters.longlive_sparse.case_identity import (
    build_case_identity,
    resolve_experiment_commit,
    validate_case_identity,
)


COMMIT = "a" * 40


def identity(**overrides):
    values = {
        "commit": COMMIT,
        "method": "block64_history",
        "prompt_id": "state_water_pour",
        "prompt": "Water rises continuously.",
        "seed": 20260826,
        "latent_frames": 120,
        "history_density": 0.25,
        "rope_policy": "upstream_zero",
        "refresh_policy": "per_chunk",
        "backend": "grouped_fa2",
    }
    values.update(overrides)
    return build_case_identity(**values)


def test_case_identity_is_stable_and_self_validating():
    first = identity()
    second = identity()
    assert first == second
    assert first["id"].endswith("k" + first["case_key_sha256"][:12])
    assert validate_case_identity({**first, **first["case_key"]}) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit", "b" * 40),
        ("prompt", "Water resets."),
        ("seed", 20260827),
        ("latent_frames", 240),
        ("history_density", 0.15),
        ("rope_policy", "recency_rank"),
        ("refresh_policy", "per_step"),
        ("backend", "fixed64_rect"),
    ],
)
def test_case_identity_changes_for_every_frozen_dimension(field, value):
    assert identity(**{field: value})["case_key_sha256"] != identity()["case_key_sha256"]


def test_full_commit_is_required():
    with pytest.raises(ValueError, match="full 40-character"):
        resolve_experiment_commit("abc", verify_checkout=False)
