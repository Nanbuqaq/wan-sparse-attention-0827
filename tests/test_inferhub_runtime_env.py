from __future__ import annotations

import os
import subprocess
from pathlib import Path


def make_bundle(root: Path) -> None:
    (root / "python-overlay").mkdir(parents=True)
    (root / "model").mkdir()
    (root / "checkpoints").mkdir()
    (root / "checkpoints/longlive_init.pt").touch()
    (root / "checkpoints/longlive_lora_003000.pt").touch()


def test_explicit_input_bundle_wins_over_legacy_weights_dir(tmp_path):
    code = tmp_path / "code"
    output = tmp_path / "output"
    venv = tmp_path / "venv"
    explicit = tmp_path / "explicit"
    legacy = tmp_path / "legacy"
    for path in (code / "third_party/longlive-inferhub", code / "third_party/LongLive-RAG", output, venv / "lib"):
        path.mkdir(parents=True)
    make_bundle(explicit)
    make_bundle(legacy)
    script = Path(__file__).resolve().parents[1] / "scripts/inferhub_runtime_env.sh"
    environment = {
        **os.environ,
        "INFER_CODE_DIR": str(code),
        "INFER_OUTPUT_DIR": str(output),
        "CUDA_VISIBLE_DEVICES": "0",
        "VIRTUAL_ENV": str(venv),
        "LONGLIVE_INPUT_BUNDLE_ROOT": str(explicit),
        "INFER_WEIGHTS_DIR": str(legacy),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n%s\\n" "$LONGLIVE_INPUT_BUNDLE_ROOT" "$LONGLIVE_WAN_MODELS_ROOT"',
            "runtime-env-test",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.splitlines() == [str(explicit), str(explicit / "model")]


def test_platform_weights_parent_resolves_input_bundle_child(tmp_path):
    code = tmp_path / "code"
    output = tmp_path / "output"
    venv = tmp_path / "venv"
    weights = tmp_path / "weights"
    for path in (code / "third_party/longlive-inferhub", code / "third_party/LongLive-RAG", output, venv / "lib"):
        path.mkdir(parents=True)
    make_bundle(weights / "input_bundle")
    script = Path(__file__).resolve().parents[1] / "scripts/inferhub_runtime_env.sh"
    environment = {
        **os.environ,
        "INFER_CODE_DIR": str(code),
        "INFER_OUTPUT_DIR": str(output),
        "CUDA_VISIBLE_DEVICES": "0",
        "VIRTUAL_ENV": str(venv),
        "INFER_WEIGHTS_DIR": str(weights),
    }
    environment.pop("LONGLIVE_INPUT_BUNDLE_ROOT", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s" "$LONGLIVE_INPUT_BUNDLE_ROOT"',
            "runtime-env-test",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout == str(weights / "input_bundle")
