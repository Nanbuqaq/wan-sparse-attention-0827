#!/usr/bin/env python3
"""Run one local command on an idle physical GPU with a cross-workstream lock."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
from pathlib import Path


def gpu_rows() -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        rows.append({"index": index, "memory": memory, "utilization": utilization})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("command required after --")
    candidates = sorted(
        [
            row
            for row in gpu_rows()
            if row["memory"] <= args.max_memory_mib
            and row["utilization"] <= args.max_utilization
        ],
        key=lambda row: (row["memory"], row["utilization"], row["index"]),
    )
    handle = None
    selected = None
    for row in candidates:
        candidate = Path(f"/tmp/wan_sparse_gpu_{row['index']}.lock").open("w")
        try:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close()
            continue
        handle, selected = candidate, row
        break
    if selected is None or handle is None:
        raise RuntimeError("no idle unlocked local GPU")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    environment["WAN_SPARSE_PHYSICAL_GPU"] = str(selected["index"])
    print(
        f"[gpu] physical={selected['index']} memory={selected['memory']}MiB "
        f"util={selected['utilization']}% command={command}",
        flush=True,
    )
    try:
        raise SystemExit(subprocess.call(command, env=environment))
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


if __name__ == "__main__":
    main()

