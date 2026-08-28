#!/usr/bin/env python3
"""Run one local LongLive command on one idle GPU under the global workflow lock."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path


DEFAULT_GLOBAL_LOCK = Path("/tmp/wan_longlive_single_gpu.lock")


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


def eligible_gpu_rows(
    rows: list[dict], *, max_memory_mib: int, max_utilization: int
) -> list[dict]:
    return sorted(
        [
            row
            for row in rows
            if row["memory"] <= max_memory_mib
            and row["utilization"] <= max_utilization
        ],
        key=lambda row: (row["memory"], row["utilization"], row["index"]),
    )


def _has_inherited_descriptor(path: Path) -> bool:
    """Detect the descriptor inherited from an outer ``flock file command``."""

    try:
        target = path.stat()
    except FileNotFoundError:
        return False
    proc_fd = Path("/proc/self/fd")
    if not proc_fd.is_dir():
        return False
    for descriptor in proc_fd.iterdir():
        try:
            current = descriptor.stat()
        except (FileNotFoundError, PermissionError):
            continue
        if (current.st_dev, current.st_ino) == (target.st_dev, target.st_ino):
            return True
    return False


@contextmanager
def global_lock(path: Path, *, wait: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_inherited_descriptor(path):
        print(f"[lock] inherited global lock {path}", flush=True)
        yield
        return
    handle = path.open("w")
    operation = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(handle, operation)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"LongLive global GPU lock is busy: {path}") from error
    print(f"[lock] acquired global lock {path}", flush=True)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("--global-lock", type=Path, default=DEFAULT_GLOBAL_LOCK)
    parser.add_argument("--wait-for-global-lock", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("command required after --")

    with global_lock(args.global_lock, wait=args.wait_for_global_lock):
        candidates = eligible_gpu_rows(
            gpu_rows(),
            max_memory_mib=args.max_memory_mib,
            max_utilization=args.max_utilization,
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
