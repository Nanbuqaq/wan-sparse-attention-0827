from __future__ import annotations

import subprocess
from pathlib import Path


def test_final_long_runner_is_four_prompt_sharded_and_shell_valid():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_final_long_confirmation.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "requires exactly four assigned GPUs" in text
    assert "--shard-axis case" in text
    assert "--latent-frames 240" in text
    assert "native_dense" in text
    assert "native_block" in text
    assert "rag_dense" in text
    assert "expected_final_long.json" in text
    assert "terminal_state_audit.json" in text
