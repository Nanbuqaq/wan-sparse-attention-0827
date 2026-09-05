from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def test_transfer_benchmark_parses_and_has_help() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/benchmark_transfer_layouts.py"
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    subprocess.run(["/usr/bin/python3", str(script), "--help"], check=True)
