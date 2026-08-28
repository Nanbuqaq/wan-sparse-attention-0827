#!/usr/bin/env python3
"""Run one command on a currently idle physical GPU without hard-coding ids."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
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
    parser.add_argument("--global-lock")
    parser.add_argument("--wait-for-global-lock", action="store_true")
    parser.add_argument("--lock-poll-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")

    global_handle = None
    if args.global_lock:
        global_path = Path(args.global_lock)
        global_handle = global_path.open("a+")
        while True:
            try:
                fcntl.flock(global_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not args.wait_for_global_lock:
                    global_handle.close()
                    raise RuntimeError(f"global GPU lock is already held: {global_path}")
                print(f"[gpu] waiting for global lock {global_path}", flush=True)
                time.sleep(max(0.1, args.lock_poll_seconds))

    handles = []
    selected = None
    try:
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
        for row in candidates:
            lock_path = Path(f"/tmp/wan_sparse_gpu_{row['index']}.lock")
            handle = lock_path.open("a+")
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
        if args.global_lock:
            environment["WAN_SPARSE_GLOBAL_GPU_LOCK"] = str(Path(args.global_lock))
        print(
            f"[gpu] physical={selected['index']} memory={selected['memory_used_mib']}MiB "
            f"util={selected['utilization']}% global_lock={args.global_lock} command={command}",
            flush=True,
        )
        raise SystemExit(subprocess.call(command, env=environment))
    finally:
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        if global_handle is not None:
            fcntl.flock(global_handle, fcntl.LOCK_UN)
            global_handle.close()


if __name__ == "__main__":
    main()
