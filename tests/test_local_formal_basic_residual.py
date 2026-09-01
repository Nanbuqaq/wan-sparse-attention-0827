from __future__ import annotations

import subprocess
from pathlib import Path


def test_local_dual_gpu_residual_scripts_are_shell_valid():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "local_formal_basic_residual_lane.sh",
        "local_formal_basic_residual_dual_gpu.sh",
    ):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
    text = (root / "scripts/local_formal_basic_residual_dual_gpu.sh").read_text()
    assert "--physical-gpu" in text
    assert "--global-lock" not in text
    assert "terminal_state_audit.json" in text
