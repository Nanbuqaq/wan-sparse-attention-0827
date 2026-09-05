"""Physical transfer plans derived from immutable history route plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch

from .route_plan import HistoryRoutePlan


_LAYOUTS = {"legacy", "exact_compact", "block64", "page256", "frame1560"}
_TRANSFER_EXECUTIONS = {"direct_multirun", "packed_separate", "packed_fused"}


@dataclass(frozen=True)
class TransferRun:
    batch_index: int
    head_index: int
    source_offset: int
    token_count: int
    destination_offset: int


@dataclass
class TransferPlan:
    """A physical layout that preserves every logical edge in a route plan."""

    route_plan_sha256: str
    layout: str
    frame_tokens: int
    page_tokens: int
    bytes_per_token: int
    candidate_frame_ids: tuple[int, ...]
    physical_source_offsets: torch.Tensor
    physical_counts: torch.Tensor
    copy_source_offsets: torch.Tensor
    copy_counts: torch.Tensor
    logical_to_physical: torch.Tensor
    resident_logical_mask: torch.Tensor
    source_runs: tuple[TransferRun, ...]
    logical_tokens: int
    resident_tokens: int
    missing_logical_tokens: int
    physical_copy_tokens: int
    padding_tokens: int

    @property
    def payload_bytes(self) -> int:
        return self.missing_logical_tokens * self.bytes_per_token

    @property
    def physical_copy_bytes(self) -> int:
        return self.physical_copy_tokens * self.bytes_per_token

    @property
    def expanded_copy_tokens(self) -> int:
        """Tokens after layout expansion but before rectangular head padding."""

        return sum(run.token_count for run in self.source_runs)

    @property
    def expanded_copy_bytes(self) -> int:
        return self.expanded_copy_tokens * self.bytes_per_token

    @property
    def granularity_padding_tokens(self) -> int:
        return max(0, self.expanded_copy_tokens - self.missing_logical_tokens)

    @property
    def rectangular_padding_tokens(self) -> int:
        return max(0, self.physical_copy_tokens - self.expanded_copy_tokens)

    @property
    def padding_bytes(self) -> int:
        return self.padding_tokens * self.bytes_per_token

    @property
    def source_run_count(self) -> int:
        return len(self.source_runs)

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.route_plan_sha256.encode())
        digest.update(self.layout.encode())
        digest.update(
            json.dumps(
                {
                    "frame_tokens": self.frame_tokens,
                    "page_tokens": self.page_tokens,
                    "bytes_per_token": self.bytes_per_token,
                    "candidate_frame_ids": self.candidate_frame_ids,
                },
                sort_keys=True,
            ).encode()
        )
        for tensor in (
            self.physical_source_offsets,
            self.physical_counts,
            self.copy_source_offsets,
            self.copy_counts,
            self.logical_to_physical,
            self.resident_logical_mask,
        ):
            cpu = tensor.detach().to("cpu").contiguous()
            digest.update(str(cpu.dtype).encode())
            digest.update(json.dumps(list(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
        for run in self.source_runs:
            digest.update(json.dumps(asdict(run), sort_keys=True).encode())
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_plan_sha256": self.route_plan_sha256,
            "transfer_plan_sha256": self.digest(),
            "layout": self.layout,
            "frame_tokens": self.frame_tokens,
            "page_tokens": self.page_tokens,
            "bytes_per_token": self.bytes_per_token,
            "candidate_frame_ids": list(self.candidate_frame_ids),
            "logical_tokens": self.logical_tokens,
            "resident_tokens": self.resident_tokens,
            "missing_logical_tokens": self.missing_logical_tokens,
            "physical_copy_tokens": self.physical_copy_tokens,
            "expanded_copy_tokens": self.expanded_copy_tokens,
            "padding_tokens": self.padding_tokens,
            "granularity_padding_tokens": self.granularity_padding_tokens,
            "rectangular_padding_tokens": self.rectangular_padding_tokens,
            "payload_bytes": self.payload_bytes,
            "physical_copy_bytes": self.physical_copy_bytes,
            "expanded_copy_bytes": self.expanded_copy_bytes,
            "padding_bytes": self.padding_bytes,
            "source_run_count": self.source_run_count,
            "physical_counts": self.physical_counts.detach().to("cpu").tolist(),
            "copy_counts": self.copy_counts.detach().to("cpu").tolist(),
            "source_runs": [asdict(run) for run in self.source_runs],
        }


@dataclass(frozen=True)
class TransferExecutionPlan:
    """How one immutable TransferPlan is physically packed and copied."""

    mode: str
    copied_tokens: int
    copied_bytes: int
    padding_bytes: int
    h2d_copy_count: int
    pack_run_count: int
    pack_bytes: int

    def __post_init__(self) -> None:
        if self.mode not in _TRANSFER_EXECUTIONS:
            raise ValueError(f"unsupported transfer execution mode: {self.mode!r}")
        for name in (
            "copied_tokens",
            "copied_bytes",
            "padding_bytes",
            "h2d_copy_count",
            "pack_run_count",
            "pack_bytes",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_transfer_execution_plan(
    transfer_plan: TransferPlan,
    *,
    mode: str,
) -> TransferExecutionPlan:
    """Account for direct runs versus packed separate/fused H2D copies.

    Direct multi-run copies layout-expanded source runs and avoids per-head
    rectangular padding. Packed modes materialize one rectangular tensor so
    they pay that padding but reduce the H2D copy count to two or one.
    """

    if mode not in _TRANSFER_EXECUTIONS:
        raise ValueError(f"unsupported transfer execution mode: {mode!r}")
    if mode == "direct_multirun":
        copied_tokens = transfer_plan.expanded_copy_tokens
        h2d_copy_count = 2 * transfer_plan.source_run_count
        pack_run_count = 0
        pack_bytes = 0
    else:
        copied_tokens = transfer_plan.physical_copy_tokens
        h2d_copy_count = 1 if mode == "packed_fused" else 2
        if copied_tokens == 0:
            h2d_copy_count = 0
        pack_run_count = transfer_plan.source_run_count
        pack_bytes = copied_tokens * transfer_plan.bytes_per_token
    copied_bytes = copied_tokens * transfer_plan.bytes_per_token
    return TransferExecutionPlan(
        mode=mode,
        copied_tokens=copied_tokens,
        copied_bytes=copied_bytes,
        padding_bytes=max(0, copied_bytes - transfer_plan.payload_bytes),
        h2d_copy_count=h2d_copy_count,
        pack_run_count=pack_run_count,
        pack_bytes=pack_bytes,
    )


def _candidate_ids(value: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        ids = tuple(int(item) for item in value.detach().to("cpu").reshape(-1))
    else:
        ids = tuple(int(item) for item in value)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("candidate_frame_ids must be non-empty and unique")
    return ids


def _expanded_offsets(
    offsets: Iterable[int], *, frame_tokens: int, granularity: int
) -> list[int]:
    expanded: set[int] = set()
    for source_offset in offsets:
        frame_index, token_id = divmod(int(source_offset), frame_tokens)
        start = (token_id // granularity) * granularity
        end = min(frame_tokens, start + granularity)
        frame_start = frame_index * frame_tokens
        expanded.update(frame_start + token for token in range(start, end))
    return sorted(expanded)


def _runs(
    offsets: list[int], *, batch_index: int, head_index: int
) -> list[TransferRun]:
    if not offsets:
        return []
    result: list[TransferRun] = []
    run_start = offsets[0]
    previous = offsets[0]
    destination = 0
    for offset in offsets[1:]:
        if offset == previous + 1:
            previous = offset
            continue
        count = previous - run_start + 1
        result.append(
            TransferRun(batch_index, head_index, run_start, count, destination)
        )
        destination += count
        run_start = previous = offset
    count = previous - run_start + 1
    result.append(
        TransferRun(batch_index, head_index, run_start, count, destination)
    )
    return result


def build_transfer_plan(
    route_plan: HistoryRoutePlan,
    candidate_frame_ids: Sequence[int] | torch.Tensor,
    *,
    frame_tokens: int,
    layout: str = "exact_compact",
    page_tokens: int = 256,
    bytes_per_token: int = 512,
    resident_logical_mask: torch.Tensor | None = None,
) -> TransferPlan:
    """Derive a deterministic physical layout without changing route semantics."""

    if layout not in _LAYOUTS:
        raise ValueError(f"unsupported transfer layout: {layout!r}")
    if frame_tokens < 1 or page_tokens < 1 or bytes_per_token < 1:
        raise ValueError("frame_tokens, page_tokens and bytes_per_token must be positive")
    ids = _candidate_ids(candidate_frame_ids)
    frame_to_rank = {frame_id: rank for rank, frame_id in enumerate(ids)}
    union_frames = route_plan.union_frame_ids.detach().to("cpu")
    union_tokens = route_plan.union_token_ids.detach().to("cpu")
    valid = (union_frames >= 0) & (union_tokens >= 0)
    if resident_logical_mask is None:
        resident = torch.zeros_like(valid)
    else:
        resident = resident_logical_mask.detach().to("cpu", dtype=torch.bool)
        if resident.shape != valid.shape:
            raise ValueError("resident_logical_mask must match route union shape")
        if bool((resident & ~valid).any()):
            raise ValueError("padded route coordinates cannot be resident")

    granularity = {
        "legacy": 1,
        "exact_compact": 1,
        "block64": 64,
        "page256": page_tokens,
        "frame1560": frame_tokens,
    }[layout]
    batch, heads, union_width = union_frames.shape
    per_head_offsets: list[list[list[int]]] = [
        [[] for _ in range(heads)] for _ in range(batch)
    ]
    per_head_maps: list[list[dict[int, int]]] = [
        [{} for _ in range(heads)] for _ in range(batch)
    ]
    per_head_copy_offsets: list[list[list[int]]] = [
        [[] for _ in range(heads)] for _ in range(batch)
    ]
    all_runs: list[TransferRun] = []
    payload_tokens = int((valid & ~resident).sum())

    for batch_index in range(batch):
        for head_index in range(heads):
            logical_offsets: list[int] = []
            missing_offsets: list[int] = []
            for union_index in range(union_width):
                if not bool(valid[batch_index, head_index, union_index]):
                    continue
                frame_id = int(union_frames[batch_index, head_index, union_index])
                token_id = int(union_tokens[batch_index, head_index, union_index])
                if frame_id not in frame_to_rank:
                    raise KeyError(
                        f"route frame {frame_id} is outside candidate_frame_ids"
                    )
                if not 0 <= token_id < frame_tokens:
                    raise IndexError(f"route token {token_id} is outside its frame")
                source_offset = frame_to_rank[frame_id] * frame_tokens + token_id
                logical_offsets.append(source_offset)
                if not bool(resident[batch_index, head_index, union_index]):
                    missing_offsets.append(source_offset)
            physical = _expanded_offsets(
                logical_offsets, frame_tokens=frame_tokens, granularity=granularity
            )
            copied = _expanded_offsets(
                missing_offsets, frame_tokens=frame_tokens, granularity=granularity
            )
            per_head_offsets[batch_index][head_index] = physical
            per_head_copy_offsets[batch_index][head_index] = copied
            per_head_maps[batch_index][head_index] = {
                offset: index for index, offset in enumerate(physical)
            }
            all_runs.extend(
                _runs(copied, batch_index=batch_index, head_index=head_index)
            )

    max_physical = max(
        (len(values) for batch_values in per_head_offsets for values in batch_values),
        default=0,
    )
    physical_tensor = torch.full((batch, heads, max_physical), -1, dtype=torch.long)
    physical_counts = torch.zeros((batch, heads), dtype=torch.long)
    max_copy = max(
        (len(values) for batch_values in per_head_copy_offsets for values in batch_values),
        default=0,
    )
    copy_tensor = torch.full((batch, heads, max_copy), -1, dtype=torch.long)
    copy_counts = torch.zeros((batch, heads), dtype=torch.long)
    logical_to_physical = torch.full_like(union_frames, -1)
    for batch_index in range(batch):
        for head_index in range(heads):
            offsets = per_head_offsets[batch_index][head_index]
            physical_counts[batch_index, head_index] = len(offsets)
            if offsets:
                physical_tensor[batch_index, head_index, : len(offsets)] = torch.tensor(
                    offsets, dtype=torch.long
                )
            copy_offsets = per_head_copy_offsets[batch_index][head_index]
            copy_counts[batch_index, head_index] = len(copy_offsets)
            if copy_offsets:
                copy_tensor[batch_index, head_index, : len(copy_offsets)] = torch.tensor(
                    copy_offsets, dtype=torch.long
                )
            mapping = per_head_maps[batch_index][head_index]
            for union_index in range(union_width):
                if not bool(valid[batch_index, head_index, union_index]):
                    continue
                frame_id = int(union_frames[batch_index, head_index, union_index])
                token_id = int(union_tokens[batch_index, head_index, union_index])
                source_offset = frame_to_rank[frame_id] * frame_tokens + token_id
                logical_to_physical[batch_index, head_index, union_index] = mapping[
                    source_offset
                ]

    physical_copy_tokens = batch * heads * max_copy
    return TransferPlan(
        route_plan_sha256=route_plan.digest(),
        layout=layout,
        frame_tokens=frame_tokens,
        page_tokens=page_tokens,
        bytes_per_token=bytes_per_token,
        candidate_frame_ids=ids,
        physical_source_offsets=physical_tensor,
        physical_counts=physical_counts,
        copy_source_offsets=copy_tensor,
        copy_counts=copy_counts,
        logical_to_physical=logical_to_physical,
        resident_logical_mask=resident,
        source_runs=tuple(all_runs),
        logical_tokens=int(valid.sum()),
        resident_tokens=int((valid & resident).sum()),
        missing_logical_tokens=payload_tokens,
        physical_copy_tokens=physical_copy_tokens,
        padding_tokens=max(0, physical_copy_tokens - payload_tokens),
    )
