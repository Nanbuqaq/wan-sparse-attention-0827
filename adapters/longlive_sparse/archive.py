"""CPU history archive, exact token materialization, and transfer accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import SparseHistoryConfig
from .rope import apply_selected_rope, build_sparse_positions
from .selectors import (
    FrameIndex,
    PretransferQuerySummary,
    SparseSelection,
    build_frame_index,
    gather_per_head,
    route_indexed_history,
    select_history,
)
from .stats import SparseRunStats


@dataclass
class MaterializedHistory:
    key_unrotated: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    transferred_bytes: int
    cpu_gather_s: float
    h2d_s: float
    rope_s: float


class HistoryArchive:
    """Per-layer archive of original unrotated K/V and retrieval metadata."""

    def __init__(self, config: SparseHistoryConfig, *, spatial_height: int, spatial_width: int):
        if spatial_height < 1 or spatial_width < 1:
            raise ValueError("spatial dimensions must be positive")
        self.config = config
        self.spatial_height = int(spatial_height)
        self.spatial_width = int(spatial_width)
        self._layers: dict[int, dict[int, FrameIndex]] = {}
        self.stats = SparseRunStats(method=config.method)

    def reset(self) -> None:
        self.clear_frames()
        self.stats = SparseRunStats(method=self.config.method)

    def clear_frames(self) -> None:
        self._layers.clear()

    def frame_ids(self, layer_id: int) -> list[int]:
        return sorted(self._layers.get(int(layer_id), {}))

    def index_frame(
        self,
        layer_id: int,
        frame_id: int,
        k_unrotated: torch.Tensor,
        v: torch.Tensor,
        *,
        storage_k: torch.Tensor | None = None,
        storage_v: torch.Tensor | None = None,
    ) -> FrameIndex:
        """Index one frame and retain references to its CPU storage tensors."""

        layer_id = int(layer_id)
        frame_id = int(frame_id)
        if k_unrotated.shape != v.shape or k_unrotated.ndim != 4:
            raise ValueError("K/V must share [B,T,H,D]")
        expected_tokens = self.spatial_height * self.spatial_width
        if k_unrotated.shape[1] != expected_tokens:
            raise ValueError(
                f"frame token count {k_unrotated.shape[1]} != spatial grid {expected_tokens}"
            )
        if frame_id in self._layers.setdefault(layer_id, {}):
            raise ValueError(f"frame {frame_id} is already indexed for layer {layer_id}")

        if storage_k is None:
            storage_k = k_unrotated.detach().to("cpu")
        if storage_v is None:
            storage_v = v.detach().to("cpu")
        if storage_k.device.type != "cpu" or storage_v.device.type != "cpu":
            raise ValueError("archive storage must reside on CPU")
        index = build_frame_index(
            frame_id,
            k_unrotated.detach(),
            storage_v,
            storage_k,
            self.config,
            spatial_height=self.spatial_height,
            spatial_width=self.spatial_width,
        )
        self._layers[layer_id][frame_id] = index
        self.stats.record_index(
            archive_bytes=index.archive_bytes,
            index_bytes=index.index_bytes,
            elapsed_s=index.index_elapsed_s,
        )
        return index

    def select(
        self,
        layer_id: int,
        query_unrotated: torch.Tensor,
        candidate_frame_ids: torch.Tensor | list[int],
    ) -> SparseSelection:
        if isinstance(candidate_frame_ids, torch.Tensor):
            ids = [int(value) for value in candidate_frame_ids.detach().cpu().reshape(-1).tolist()]
        else:
            ids = [int(value) for value in candidate_frame_ids]
        if not ids:
            raise ValueError("candidate_frame_ids cannot be empty")
        layer = self._layers.get(int(layer_id), {})
        missing = [frame_id for frame_id in ids if frame_id not in layer]
        if missing:
            raise KeyError(f"unindexed history frames for layer {layer_id}: {missing}")
        return select_history(query_unrotated, [layer[frame_id] for frame_id in ids], self.config)

    def route_indexed(
        self,
        layer_id: int,
        query_unrotated: torch.Tensor | PretransferQuerySummary,
        candidate_frame_ids: torch.Tensor | list[int],
        *,
        exact_k_tokens: int,
    ):
        if isinstance(candidate_frame_ids, torch.Tensor):
            ids = [
                int(value)
                for value in candidate_frame_ids.detach().cpu().reshape(-1).tolist()
            ]
        else:
            ids = [int(value) for value in candidate_frame_ids]
        layer = self._layers.get(int(layer_id), {})
        missing = [frame_id for frame_id in ids if frame_id not in layer]
        if missing:
            raise KeyError(f"unindexed history frames for layer {layer_id}: {missing}")
        return route_indexed_history(
            query_unrotated,
            [layer[frame_id] for frame_id in ids],
            self.config,
            exact_k_tokens=exact_k_tokens,
        )

    def materialize(
        self,
        layer_id: int,
        selection: SparseSelection,
        *,
        device: torch.device | str,
        current_frame_id: int,
        freqs: torch.Tensor | None,
        candidate_frame_ids: torch.Tensor | list[int] | None = None,
        dense_key: torch.Tensor | None = None,
        dense_value: torch.Tensor | None = None,
        dense_frame_ids: torch.Tensor | None = None,
        dense_token_ids: torch.Tensor | None = None,
    ) -> MaterializedHistory:
        """Gather original K/V, transfer selected bytes, and apply sparse RoPE."""

        target_device = torch.device(device)
        frames = self._layers[int(layer_id)]
        frame_ids = selection.frame_ids
        token_ids = selection.token_ids
        batch, heads, selected_tokens = frame_ids.shape
        first = next(iter(frames.values()))
        dim = first.key.shape[-1]
        dtype = first.key.dtype
        use_pinned = bool(
            self.config.pin_memory
            and target_device.type == "cuda"
            and torch.cuda.is_available()
        )

        gather_start = time.perf_counter()
        dense_arguments = (dense_key, dense_value, dense_frame_ids, dense_token_ids)
        if any(value is not None for value in dense_arguments):
            if not all(value is not None for value in dense_arguments):
                raise ValueError("dense materialization requires key/value/frame/token tensors")
            if dense_key.device.type != "cpu" or dense_value.device.type != "cpu":
                raise ValueError("dense materialization source must be on CPU")
            valid = frame_ids >= 0
            max_token = max(
                int(dense_token_ids.max()) if dense_token_ids.numel() else 0,
                int(token_ids[valid].max()) if valid.any() else 0,
            )
            base = max_token + 1
            dense_codes = dense_frame_ids.long() * base + dense_token_ids.long()
            selected_codes = frame_ids.long() * base + token_ids.clamp_min(0).long()
            sorted_codes, sorted_to_dense = torch.sort(dense_codes, dim=-1)
            sorted_indices = torch.searchsorted(
                sorted_codes.contiguous(), selected_codes.contiguous()
            ).clamp_max(dense_codes.shape[-1] - 1)
            dense_indices = sorted_to_dense.gather(-1, sorted_indices)
            matched = dense_codes.gather(-1, dense_indices) == selected_codes
            if not bool((matched | ~valid).all()):
                raise KeyError("selection contains coordinates outside dense candidate tensors")
            dense_indices = torch.where(valid, dense_indices, torch.zeros_like(dense_indices))
            gathered_key = gather_per_head(dense_key, dense_indices)
            gathered_value = gather_per_head(dense_value, dense_indices)
            if not bool(valid.all()):
                mask = valid.permute(0, 2, 1).unsqueeze(-1)
                gathered_key = gathered_key.masked_fill(~mask, 0)
                gathered_value = gathered_value.masked_fill(~mask, 0)
            if use_pinned:
                key_cpu = torch.empty_like(gathered_key, pin_memory=True)
                value_cpu = torch.empty_like(gathered_value, pin_memory=True)
                key_cpu.copy_(gathered_key)
                value_cpu.copy_(gathered_value)
            else:
                key_cpu = gathered_key
                value_cpu = gathered_value
        else:
            key_bhkd = torch.zeros(
                (batch, heads, selected_tokens, dim),
                dtype=dtype,
                device="cpu",
                pin_memory=use_pinned,
            )
            value_bhkd = torch.zeros_like(key_bhkd, pin_memory=use_pinned)
            valid = (frame_ids >= 0) & (token_ids >= 0)
            for frame_id, frame in frames.items():
                locations = torch.nonzero(
                    valid & (frame_ids == frame_id), as_tuple=False
                )
                if not locations.numel():
                    continue
                batch_ids, head_ids, output_ids = locations.unbind(dim=1)
                source_tokens = token_ids[batch_ids, head_ids, output_ids]
                source_key = frame.key.permute(0, 2, 1, 3)
                source_value = frame.value.permute(0, 2, 1, 3)
                key_bhkd[batch_ids, head_ids, output_ids] = source_key[
                    batch_ids, head_ids, source_tokens
                ]
                value_bhkd[batch_ids, head_ids, output_ids] = source_value[
                    batch_ids, head_ids, source_tokens
                ]
            known_frames = torch.tensor(list(frames), dtype=frame_ids.dtype)
            if bool(valid.any()) and not bool(
                torch.isin(frame_ids[valid], known_frames).all()
            ):
                raise KeyError("selection contains a frame outside the CPU archive")
            key_cpu = key_bhkd.permute(0, 2, 1, 3).contiguous()
            value_cpu = value_bhkd.permute(0, 2, 1, 3).contiguous()
        cpu_gather_s = time.perf_counter() - gather_start

        transfer_start = time.perf_counter()
        key_device = key_cpu.to(
            target_device,
            non_blocking=use_pinned and self.config.non_blocking_h2d,
        )
        value_device = value_cpu.to(
            target_device,
            non_blocking=use_pinned and self.config.non_blocking_h2d,
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        h2d_s = time.perf_counter() - transfer_start

        if candidate_frame_ids is None:
            candidate_tensor = None
        elif isinstance(candidate_frame_ids, torch.Tensor):
            candidate_tensor = candidate_frame_ids
        else:
            candidate_tensor = torch.tensor(candidate_frame_ids, dtype=torch.long)
        positions = build_sparse_positions(
            frame_ids=frame_ids.clamp_min(0),
            token_ids=token_ids.clamp_min(0),
            current_frame_id=current_frame_id,
            spatial_width=self.spatial_width,
            rope_policy=self.config.rope_policy,
            max_relative_age=self.config.max_relative_age,
            candidate_frame_ids=candidate_tensor,
        )
        rope_start = time.perf_counter()
        key_unrotated_device = key_device
        if freqs is not None:
            key_device = apply_selected_rope(
                key_unrotated_device,
                positions.to(target_device),
                freqs.to(target_device),
            )
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
        rope_s = time.perf_counter() - rope_start
        transferred_bytes = (key_device.numel() + value_device.numel()) * key_device.element_size()
        return MaterializedHistory(
            key_unrotated=key_unrotated_device,
            key=key_device,
            value=value_device,
            positions=positions,
            transferred_bytes=transferred_bytes,
            cpu_gather_s=cpu_gather_s,
            h2d_s=h2d_s,
            rope_s=rope_s,
        )

    def archive_bytes(self) -> int:
        return sum(
            frame.archive_bytes
            for layer in self._layers.values()
            for frame in layer.values()
        )

    def candidate_history_size(
        self, layer_id: int, frame_ids: torch.Tensor | list[int]
    ) -> tuple[int, int]:
        if isinstance(frame_ids, torch.Tensor):
            ids = [int(value) for value in frame_ids.detach().cpu().reshape(-1)]
        else:
            ids = [int(value) for value in frame_ids]
        layer = self._layers.get(int(layer_id), {})
        missing = [frame_id for frame_id in ids if frame_id not in layer]
        if missing:
            raise KeyError(f"unindexed history frames for layer {layer_id}: {missing}")
        frames = [layer[frame_id] for frame_id in ids]
        tokens = sum(frame.key.shape[1] for frame in frames)
        candidate_bytes = sum(
            (frame.key.numel() + frame.value.numel()) * frame.key.element_size()
            for frame in frames
        )
        return tokens, candidate_bytes

    def dense_history_tensors(
        self,
        layer_id: int,
        candidate_frame_ids: torch.Tensor | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full candidate history on CPU with original coordinates."""

        if isinstance(candidate_frame_ids, torch.Tensor):
            ids = [int(value) for value in candidate_frame_ids.detach().cpu().reshape(-1).tolist()]
        else:
            ids = [int(value) for value in candidate_frame_ids]
        layer = self._layers.get(int(layer_id), {})
        missing = [frame_id for frame_id in ids if frame_id not in layer]
        if missing:
            raise KeyError(f"unindexed history frames for layer {layer_id}: {missing}")
        frames = [layer[frame_id] for frame_id in ids]
        key = torch.cat([frame.key for frame in frames], dim=1)
        value = torch.cat([frame.value for frame in frames], dim=1)
        batch, tokens, heads, _ = key.shape
        per_frame = frames[0].key.shape[1]
        frame_ids = torch.tensor(ids, dtype=torch.long).repeat_interleave(per_frame)
        token_ids = torch.arange(per_frame, dtype=torch.long).repeat(len(ids))
        return (
            key,
            value,
            frame_ids.view(1, 1, tokens).expand(batch, heads, -1).clone(),
            token_ids.view(1, 1, tokens).expand(batch, heads, -1).clone(),
        )

    def index_bytes(self) -> int:
        return sum(
            frame.index_bytes
            for layer in self._layers.values()
            for frame in layer.values()
        )
