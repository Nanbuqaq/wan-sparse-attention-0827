from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_training_gate_stays_closed_before_all_conditions(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "status": "pass",
                "expected_cases": 44,
                "terminal_cases": 44,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "multiple_correct_methods": True,
                "multiple_long_videos": True,
                "stable_late_degradation": True,
                "density_50_improves": False,
                "refresh_rope_backend_exhausted": True,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "decision.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/audit_training_gate.py"),
            "--base-audit",
            str(base),
            "--diagnostics",
            str(diagnostics),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    decision = json.loads(output.read_text())
    assert decision["decision"] == "do_not_train"
    assert decision["training_triggered"] is False
