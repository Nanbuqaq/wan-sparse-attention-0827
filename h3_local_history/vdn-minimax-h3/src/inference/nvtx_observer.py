"""Opt-in NVTX ranges for H3 inference profiling.

The observer is deliberately inert unless ``VDN_NVTX=1``.  It never synchronizes,
changes tensor values, or changes control flow; Nsight Systems can therefore attribute
CUDA launches to the CPU ranges without turning the normal runner into a profiled one.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch


ENABLED = os.environ.get("VDN_NVTX", "0").lower() in {"1", "true", "yes", "on"}


@contextmanager
def range_(label: str):
    if not ENABLED:
        yield
        return
    pushed = False
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(label)
            pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def enabled() -> bool:
    return ENABLED
