"""Canonical, commit-aware identities for formal LongLive video cases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: object, *, limit: int = 48) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-_")
    return (normalized or "none")[:limit]


def _density_token(value: float) -> str:
    normalized = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return normalized.replace("-", "m").replace(".", "p")


def resolve_experiment_commit(
    explicit: str | None = None,
    *,
    repo_root: str | Path | None = None,
    verify_checkout: bool = True,
) -> str:
    """Resolve a full commit SHA and optionally require it to match the checkout."""

    requested = explicit or os.environ.get("LONGLIVE_EXPERIMENT_COMMIT")
    checkout = None
    if repo_root is not None:
        checkout = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    commit = requested or checkout
    if commit is None or not _FULL_COMMIT.fullmatch(commit):
        raise ValueError(f"formal experiment requires a full 40-character commit SHA: {commit!r}")
    if verify_checkout and checkout is not None and checkout != commit:
        raise ValueError(
            f"experiment commit {commit} does not match checkout HEAD {checkout}"
        )
    return commit


def resolve_experiment_provenance(
    explicit: str | None = None,
    *,
    repo_root: str | Path,
) -> tuple[str, str, str]:
    """Resolve scientific case identity and the actual execution checkout.

    A different execution checkout is allowed only for a declared harness-only
    change.  This keeps attention/model identities comparable while making the
    post-generation implementation delta explicit in every state artifact.
    """

    execution_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if not _FULL_COMMIT.fullmatch(execution_commit):
        raise ValueError(
            f"execution checkout did not resolve to a full commit: {execution_commit!r}"
        )
    requested = explicit or os.environ.get("LONGLIVE_EXPERIMENT_COMMIT")
    scope = os.environ.get("LONGLIVE_EXECUTION_CHANGE_SCOPE", "").strip()
    if requested and requested != execution_commit:
        if not scope:
            raise ValueError(
                "experiment/execution commit mismatch requires "
                "LONGLIVE_EXECUTION_CHANGE_SCOPE"
            )
        experiment_commit = resolve_experiment_commit(
            requested,
            repo_root=repo_root,
            verify_checkout=False,
        )
    else:
        experiment_commit = execution_commit
        scope = scope or "same_checkout"
    return experiment_commit, execution_commit, scope


def build_case_identity(
    *,
    commit: str,
    method: str,
    prompt_id: str,
    prompt: str,
    seed: int,
    latent_frames: int,
    history_density: float,
    rope_policy: str,
    refresh_policy: str,
    backend: str,
    system_identity: Mapping[str, Any] | None = None,
    method_params: Mapping[str, Any] | None = None,
) -> dict:
    """Return the canonical case key, digest and filesystem-safe case id."""

    if not _FULL_COMMIT.fullmatch(commit):
        raise ValueError("case identity requires a full lowercase commit SHA")
    density = float(history_density)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"history_density must be in [0, 1], got {density}")
    if int(latent_frames) <= 0:
        raise ValueError("latent_frames must be positive")
    case_key = {
        "schema_version": 3 if method_params is not None else (2 if system_identity is not None else 1),
        "commit": commit,
        "method": str(method),
        "prompt_id": str(prompt_id),
        "prompt_sha256": sha256_text(str(prompt)),
        "seed": int(seed),
        "latent_frames": int(latent_frames),
        "history_density": density,
        "rope_policy": str(rope_policy),
        "refresh_policy": str(refresh_policy),
        "backend": str(backend),
    }
    if system_identity is not None:
        case_key["system"] = json.loads(
            json.dumps(dict(system_identity), ensure_ascii=False, sort_keys=True)
        )
    if method_params is not None:
        case_key["method_params"] = json.loads(json.dumps(dict(method_params), sort_keys=True))
    canonical = json.dumps(
        case_key, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    digest = sha256_text(canonical)
    parts = [
            _slug(method),
            _slug(prompt_id),
            f"s{int(seed)}",
            f"lf{int(latent_frames)}",
            f"d{_density_token(density)}",
            f"rope-{_slug(rope_policy)}",
            f"refresh-{_slug(refresh_policy)}",
            f"backend-{_slug(backend)}",
    ]
    if system_identity is not None:
        system_digest = sha256_text(
            json.dumps(
                case_key["system"],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        parts.append(f"sys-{system_digest[:10]}")
    parts.extend((f"c{commit[:12]}", f"k{digest[:12]}"))
    case_id = "__".join(parts)
    return {
        "id": case_id,
        "case_id": case_id,
        "case_key": case_key,
        "case_key_sha256": digest,
        "commit": commit,
    }


def validate_case_identity(record: dict) -> list[str]:
    """Return validation errors for a serialized formal case record."""

    key = record.get("case_key")
    if not isinstance(key, dict):
        return ["missing case_key"]
    required = {
        "schema_version",
        "commit",
        "method",
        "prompt_id",
        "prompt_sha256",
        "seed",
        "latent_frames",
        "history_density",
        "rope_policy",
        "refresh_policy",
        "backend",
    }
    missing = sorted(required - set(key))
    if missing:
        return [f"case_key missing fields: {missing}"]
    schema_version = int(key.get("schema_version", 0))
    if schema_version not in {1, 2, 3}:
        return [f"unsupported case identity schema: {schema_version}"]
    if schema_version == 2 and not isinstance(key.get("system"), dict):
        return ["case_key schema v2 requires system mapping"]
    if schema_version == 3 and not isinstance(key.get("method_params"), dict):
        return ["case_key schema v3 requires method_params mapping"]
    canonical = json.dumps(
        key, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    digest = sha256_text(canonical)
    errors = []
    if record.get("case_key_sha256") != digest:
        errors.append("case_key_sha256 mismatch")
    if record.get("commit") != key.get("commit"):
        errors.append("record commit does not match case_key")
    for field in (
        "method",
        "prompt_id",
        "seed",
        "latent_frames",
        "history_density",
        "rope_policy",
        "refresh_policy",
        "backend",
    ):
        if field in record and record[field] != key[field]:
            errors.append(f"record {field} does not match case_key")
    if "system" in record and record["system"] != key.get("system"):
        errors.append("record system does not match case_key")
    if schema_version == 3 and "method_params" in record and record["method_params"] != key.get("method_params"):
        errors.append("record method_params does not match case_key")
    return errors
