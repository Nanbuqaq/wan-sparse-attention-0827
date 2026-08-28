#!/usr/bin/env python3
"""Audit Stage-3 isolation, single-GPU policy, and live CUDA availability."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import torch


LOCK = Path("/tmp/wan_short_stage3_single_gpu.lock")
STAGE2_AUDIT = ROOT / "results/manifests/final_audit_v2.json"
STAGE3_CONFIGS = [
    ROOT / "configs/stage3_smoke_4step.json",
    ROOT / "configs/stage3_calibration_50step.json",
    ROOT / "configs/stage3_backend_100_50step.json",
    ROOT / "configs/stage3_formal_50step.template.json",
]


def main() -> None:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    LOCK.touch(exist_ok=True)
    handle = LOCK.open("a+")
    lock_available = True
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_available = False
    finally:
        if lock_available:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
    config_rows = []
    for path in STAGE3_CONFIGS:
        payload = json.loads(path.read_text())
        policy = payload.get("stage3_gpu_policy") or {}
        config_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "output_root": payload.get("output_root"),
                "manifest_root": payload.get("manifest_root"),
                "policy": policy,
                "independent_from_stage2": "stage3" in str(payload.get("output_root", "")) and "stage3" in str(payload.get("manifest_root", "")),
                "single_shard": policy.get("num_shards") == 1 and policy.get("shard_index") == 0,
                "global_lock_correct": policy.get("global_lock") == str(LOCK),
            }
        )
    lock_owned_by_runner = os.environ.get("WAN_SPARSE_GLOBAL_GPU_LOCK") == str(LOCK)
    checks = {
        "stage2_audit_still_present_and_passed": STAGE2_AUDIT.is_file() and json.loads(STAGE2_AUDIT.read_text()).get("status") == "pass",
        "stage3_configs_independent": all(row["independent_from_stage2"] for row in config_rows),
        "stage3_configs_single_shard": all(row["single_shard"] for row in config_rows),
        "stage3_configs_use_global_lock": all(row["global_lock_correct"] for row in config_rows),
        "global_lock_policy_active": lock_available or lock_owned_by_runner,
        "nvidia_smi_ok": smi.returncode == 0,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count_positive": torch.cuda.device_count() > 0,
    }
    gpu_ready = checks["nvidia_smi_ok"] and checks["torch_cuda_available"] and checks["torch_cuda_device_count_positive"]
    payload = {
        "schema_version": 3,
        "status": "ready" if gpu_ready and all(value for key, value in checks.items() if not key.startswith(("nvidia", "torch"))) else "blocked_gpu" if not gpu_ready else "fail",
        "checks": checks,
        "nvidia_smi": {"returncode": smi.returncode, "stdout": smi.stdout.strip(), "stderr": smi.stderr.strip()},
        "torch": {"version": torch.__version__, "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()},
        "configs": config_rows,
        "global_lock": str(LOCK),
        "global_lock_owned_by_runner": lock_owned_by_runner,
        "blocking_scope": "GPU validation, captured routing, smoke, calibration, and formal video generation only",
    }
    output = ROOT / "results/manifests/stage3/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
