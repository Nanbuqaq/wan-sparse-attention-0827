from __future__ import annotations

import pandas as pd

from scripts.summarize_video_cases import audit_eligibility


RULE = {
    "eligibility": {
        "required_terminal_basic_cases": 2,
        "allowed_case_statuses": ["pass", "negative"],
    }
}


def row(method, case_id, status="pass", commit="a" * 40, prompt_id="p", seed=1):
    return {
        "method": method,
        "baseline_method": "rag_dense" if method != "native_block" else "native_dense",
        "case_id": case_id,
        "status": status,
        "commit": commit,
        "prompt_id": prompt_id,
        "seed": seed,
        "latent_frames": 120,
    }


def test_pareto_requires_two_distinct_technically_valid_basic_cases():
    table = pd.DataFrame(
        [
            row("rag_dense", "d1", prompt_id="p1"),
            row("rag_dense", "d2", prompt_id="p2"),
            row("good", "g1", prompt_id="p1"),
            row("good", "g2", status="negative", prompt_id="p2"),
            row("missing", "m1"),
            row("failed", "f1", prompt_id="p1"),
            row("failed", "f2", status="fail", prompt_id="p2"),
        ]
    )
    eligible, audit = audit_eligibility(table, RULE)
    assert eligible == {"good"}
    reasons = {item["method"]: item["reasons"] for item in audit}
    assert reasons["missing"]
    assert any("invalid statuses" in reason for reason in reasons["failed"])
