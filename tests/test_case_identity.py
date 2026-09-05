from __future__ import annotations

from pathlib import Path

import pytest

from adapters.longlive_sparse.case_identity import (
    build_case_identity,
    resolve_experiment_commit,
    resolve_experiment_provenance,
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
        (
            "system_identity",
            {"transfer_layout": "block64", "gpu_union_cache_budget_mib": 768},
        ),
    ],
)
def test_case_identity_changes_for_every_frozen_dimension(field, value):
    assert identity(**{field: value})["case_key_sha256"] != identity()["case_key_sha256"]


def test_full_commit_is_required():
    with pytest.raises(ValueError, match="full 40-character"):
        resolve_experiment_commit("abc", verify_checkout=False)


def test_utility_candidates_cannot_alias_case_key():
    peak = identity(method="system_utility_history", system_identity={},
                    method_params={"value_candidate": "peak_value"})
    count = identity(method="system_utility_history", system_identity={},
                     method_params={"value_candidate": "count_uniform"})
    assert peak["case_key_sha256"] != count["case_key_sha256"]
    assert peak["case_key"]["schema_version"] == 3


def test_case_identity_v2_serializes_system_dimensions():
    system = {
        "transfer_layout": "exact_compact",
        "offload_overlap": "d2h_compute",
        "onload_overlap": "kv_stream",
    }
    result = identity(system_identity=system)
    assert result["case_key"]["schema_version"] == 2
    assert result["case_key"]["system"] == system
    assert "__sys-" in result["id"]
    assert validate_case_identity({**result, **result["case_key"]}) == []


def test_harness_only_commit_mismatch_requires_explicit_scope(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    execution = resolve_experiment_commit(repo_root=root)
    experiment = "a" * 40 if execution != "a" * 40 else "b" * 40
    monkeypatch.delenv("LONGLIVE_EXECUTION_CHANGE_SCOPE", raising=False)
    with pytest.raises(ValueError, match="requires LONGLIVE_EXECUTION_CHANGE_SCOPE"):
        resolve_experiment_provenance(experiment, repo_root=root)
    monkeypatch.setenv(
        "LONGLIVE_EXECUTION_CHANGE_SCOPE", "vae_chunk_cache_continuity_only"
    )
    assert resolve_experiment_provenance(experiment, repo_root=root) == (
        experiment,
        execution,
        "vae_chunk_cache_continuity_only",
    )


def test_execution_checkout_ignores_experiment_commit_environment(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    execution = resolve_experiment_commit(repo_root=root)
    experiment = "a" * 40 if execution != "a" * 40 else "b" * 40
    monkeypatch.setenv("LONGLIVE_EXPERIMENT_COMMIT", experiment)
    monkeypatch.setenv(
        "LONGLIVE_EXECUTION_CHANGE_SCOPE", "vae_chunk_cache_continuity_only"
    )
    assert resolve_experiment_provenance(repo_root=root) == (
        experiment,
        execution,
        "vae_chunk_cache_continuity_only",
    )
