#!/usr/bin/env python3
"""Run one command on a currently idle physical GPU without hard-coding ids."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path


def gpu_rows() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        rows.append({"index": index, "memory_used_mib": memory, "utilization": utilization})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")

    candidates = sorted(
        (
            row
            for row in gpu_rows()
            if row["memory_used_mib"] <= args.max_memory_mib
            and row["utilization"] <= args.max_utilization
        ),
        key=lambda row: (row["memory_used_mib"], row["utilization"], row["index"]),
    )
    if not candidates:
        raise RuntimeError("no GPU satisfies the idle thresholds")

    handles = []
    selected = None
    for row in candidates:
        lock_path = Path(f"/tmp/wan_sparse_gpu_{row['index']}.lock")
        handle = lock_path.open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        handles.append(handle)
        selected = row
        break
    if selected is None:
        raise RuntimeError("all idle GPUs are locked by another workstream task")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    environment["WAN_SPARSE_PHYSICAL_GPU"] = str(selected["index"])
    print(
        f"[gpu] physical={selected['index']} memory={selected['memory_used_mib']}MiB "
        f"util={selected['utilization']}% command={command}",
        flush=True,
    )
    try:
        raise SystemExit(subprocess.call(command, env=environment))
    finally:
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


if __name__ == "__main__":
    main()

