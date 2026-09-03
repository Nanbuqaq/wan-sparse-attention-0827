from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_recovery_filter_selects_only_matching_terminal_failures(tmp_path):
    expected = tmp_path / "expected.json"
    states = tmp_path / "states.json"
    output = tmp_path / "filtered.json"
    cases = [
        {"id": "a", "case_key_sha256": "ka"},
        {"id": "b", "case_key_sha256": "kb"},
        {"id": "c", "case_key_sha256": "kc"},
    ]
    expected.write_text(json.dumps({"commit": "x", "cases": cases}))
    states.write_text(
        json.dumps(
            {
                "cases": [
                    {**cases[0], "status": "pass"},
                    {
                        **cases[1],
                        "status": "fail",
                        "failure_reason": "decoded frame count 954 != expected 957",
                    },
                    {
                        **cases[2],
                        "status": "fail",
                        "failure_reason": "out of memory",
                    },
                ]
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/filter_expected_from_states.py"),
            "--expected",
            str(expected),
            "--states",
            str(states),
            "--status",
            "fail",
            "--failure-contains",
            "decoded frame count 954",
            "--expected-count",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )
    selected = json.loads(output.read_text())["cases"]
    assert [case["id"] for case in selected] == ["b"]
