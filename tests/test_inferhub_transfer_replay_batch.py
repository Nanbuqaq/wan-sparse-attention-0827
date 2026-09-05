from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def test_transfer_replay_batch_is_eight_distinct_gpu_lanes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_transfer_replay_8gpu.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "ne 8" in text
    assert "for lane in 0 1 2 3 4 5 6 7" in text
    assert "OMP_NUM_THREADS=2" in text
    assert "benchmark_transfer_layouts.py" in text
    captures = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("layer") and line.strip().endswith(".pt")
    ]
    assert len(captures) == 8
    assert len(set(captures)) == 8


def test_transfer_benchmark_and_calibrator_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("benchmark_transfer_layouts.py", "calibrate_system_cost_model.py"):
        path = root / "scripts" / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
