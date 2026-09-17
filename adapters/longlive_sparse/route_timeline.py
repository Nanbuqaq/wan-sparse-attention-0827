"""Fixed-route timeline helpers for real 5B LongLive2 windows.

The recorder deliberately separates host submission time from CUDA event service
intervals. CUDA events are synchronized only when the runner exports the audit,
after the generation wall has already been measured.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import torch


def stable_metadata_sha256(value: Any) -> str:
    """Hash route metadata without reading tensor payloads."""

    def normalize(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return {
                "shape": list(item.shape),
                "stride": list(item.stride()),
                "dtype": str(item.dtype),
                "device": str(item.device),
                "numel": int(item.numel()),
                "nbytes": int(item.numel() * item.element_size()),
                "is_contiguous": bool(item.is_contiguous()),
            }
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(x) for x in item]
        return repr(item)

    encoded = json.dumps(normalize(value), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    """Return non-payload identity and storage accounting for a tensor."""

    storage = tensor.untyped_storage()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": int(tensor.data_ptr()),
        "storage_data_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "numel": int(tensor.numel()),
        "nbytes": int(tensor.numel() * tensor.element_size()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "is_view": bool(tensor.data_ptr() != storage.data_ptr() or tensor.numel() * tensor.element_size() != storage.nbytes()),
    }


def new_cuda_events(count: int, device: torch.device) -> list[torch.cuda.Event]:
    if device.type != "cuda":
        raise ValueError("route timeline CUDA events require a CUDA tensor")
    if count < 2:
        raise ValueError("at least start and end events are required")
    return [torch.cuda.Event(enable_timing=True) for _ in range(count)]


def attach_event_series(record: dict[str, Any], name: str, events: Iterable[torch.cuda.Event]) -> None:
    series = record.setdefault("_event_series", {})
    if name in series:
        raise ValueError(f"duplicate route timeline event series: {name}")
    series[name] = list(events)


def finalize_event_series(record: dict[str, Any]) -> None:
    """Synchronize exported events and replace them with service durations."""

    series = record.pop("_event_series", {})
    for name, events in series.items():
        if len(events) < 2:
            raise ValueError(f"timeline series {name} is incomplete")
        for event in events:
            event.synchronize()
        intervals = [
            events[index].elapsed_time(events[index + 1]) / 1000.0
            for index in range(len(events) - 1)
        ]
        record[f"{name}_device_s"] = events[0].elapsed_time(events[-1]) / 1000.0
        record[f"{name}_interval_device_s"] = intervals


def finalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        finalize_event_series(record)
    return records
