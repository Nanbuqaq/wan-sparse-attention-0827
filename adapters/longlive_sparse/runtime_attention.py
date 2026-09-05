"""LongLive-RAG self-attention with training-free sparse history materialization.

Cache/RoPE control flow is adapted from the pinned LongLive-RAG repository at
commit 973884a3, whose repository-level LICENSE is Apache-2.0. Modifications,
source hashes and the upstream header discrepancy are recorded separately.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from .archive import HistoryArchive, MaterializedHistory
from .ar_routing import route_history
from .backends import execute_plan
from .config import SparseHistoryConfig
from .history_cache import (
    CachedHistoryKV,
    HistoryKVCacheKey,
    HistoryUnionCache,
    RawHistoryBlockCache,
    tensor_sha256,
)
from .methods import method_spec
from .rope import apply_selected_rope, build_sparse_positions
from .route_plan import map_union_coordinates
from .selectors import SparseSelection, gather_per_head, select_block64_from_tensor
from .selectors import (
    INDEXED_PRETRANSFER_METHODS,
    SUMMARY_PRETRANSFER_METHODS,
    summarize_query_for_pretransfer,
)
from .stats import SparseCallRecord, TimingBreakdown
from .staging import PinnedStagingPool
from .system_config import LongLiveSystemConfig
from .transfer_plan import build_transfer_plan
from .upstreams import load_latentmem_module


if torch.cuda.is_available():
    _UPSTREAM = load_latentmem_module()
    _BaseSelfAttention = _UPSTREAM.CausalWanSelfAttention
    causal_online_rope = _UPSTREAM.causal_online_rope
    from wan.modules.attention import attention, attention_backend  # noqa: E402
else:
    class _BaseSelfAttention(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "LongLive runtime attention requires a CUDA-enabled process; "
                "the selector/archive modules remain CPU-testable"
            )

    def causal_online_rope(*args: Any, **kwargs: Any):
        raise RuntimeError("causal_online_rope requires the CUDA LongLive runtime")

    def attention(*args: Any, **kwargs: Any):
        raise RuntimeError("LongLive attention requires the CUDA LongLive runtime")

    def attention_backend() -> str:
        return "cuda-unavailable"


def _timed_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    start = time.perf_counter()
    output = attention(query, key, value)
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    return output, time.perf_counter() - start


class SparseHistorySelfAttention(_BaseSelfAttention):
    """Drop-in replacement for LongLive-RAG's per-layer self-attention."""

    def __init__(
        self,
        *args: Any,
        layer_id: int,
        history_archive: HistoryArchive,
        sparse_config: SparseHistoryConfig,
        system_config: LongLiveSystemConfig | None = None,
        history_union_cache: HistoryUnionCache | RawHistoryBlockCache | None = None,
        history_staging_pool: PinnedStagingPool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layer_id = int(layer_id)
        self.history_archive = history_archive
        self.sparse_config = sparse_config
        self.system_config = system_config or LongLiveSystemConfig()
        self.history_union_cache = history_union_cache
        self.history_staging_pool = history_staging_pool
        self._selection_cache: dict[tuple[Any, ...], Any] = {}
        self._captured_qkv: set[tuple[int, int]] = set()
        self._capture_counts: dict[int, int] = {}
        self._capture_marker_counts: dict[tuple[int, int], int] = {}
        self._route_capture_counts: dict[tuple[int, int], int] = {}

    def clear_selection_cache(self) -> None:
        self._selection_cache.clear()

    def clear_capture_state(self) -> None:
        self._captured_qkv.clear()
        self._capture_counts.clear()
        self._capture_marker_counts.clear()
        self._route_capture_counts.clear()

    @staticmethod
    def _capture_root(kind: str) -> Path:
        root = Path(os.environ.get("INFER_OUTPUT_DIR", "results/captures")) / kind
        tag = os.environ.get("LONGLIVE_CAPTURE_CASE_TAG", "").strip()
        return root / tag if tag else root

    def _capture_qkv_once(
        self,
        *,
        current_start: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        frame_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        if os.environ.get("LONGLIVE_CAPTURE_QKV", "0") != "1":
            return
        layers = {
            int(item)
            for item in os.environ.get("LONGLIVE_CAPTURE_LAYERS", "0,9,19,29").split(",")
            if item.strip()
        }
        requested_starts = {
            int(item)
            for item in os.environ.get("LONGLIVE_CAPTURE_STARTS", "").split(",")
            if item.strip()
        }
        marker = (self.layer_id, int(current_start))
        per_start_limit = int(
            os.environ.get("LONGLIVE_CAPTURE_PASSES_PER_START", "1")
        )
        marker_count = self._capture_marker_counts.get(marker, 0)
        if (
            self.layer_id not in layers
            or (requested_starts and int(current_start) not in requested_starts)
            or marker_count >= per_start_limit
        ):
            return
        max_per_layer = int(os.environ.get("LONGLIVE_CAPTURE_MAX_PER_LAYER", "0"))
        if max_per_layer > 0 and self._capture_counts.get(self.layer_id, 0) >= max_per_layer:
            return
        output_root = self._capture_root("qkv_captures")
        output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "layer": self.layer_id,
            "current_start": int(current_start),
            "denoising_pass": marker_count,
            "query": query.detach().to("cpu"),
            "key": key.detach().to("cpu"),
            "value": value.detach().to("cpu"),
            "frame_ids": frame_ids.detach().to("cpu"),
            "token_ids": token_ids.detach().to("cpu"),
        }
        filename = f"layer{self.layer_id:02d}_start{int(current_start):08d}.pt"
        if per_start_limit > 1:
            filename = (
                f"layer{self.layer_id:02d}_start{int(current_start):08d}"
                f"_pass{marker_count:02d}.pt"
            )
        torch.save(
            payload,
            output_root / filename,
        )
        self._capture_marker_counts[marker] = marker_count + 1
        if per_start_limit == 1:
            self._captured_qkv.add(marker)
        self._capture_counts[self.layer_id] = self._capture_counts.get(self.layer_id, 0) + 1

    def _capture_route_reuse(
        self,
        *,
        current_start: int,
        route_plan,
        transfer_plan,
        materialized,
    ) -> None:
        if os.environ.get("LONGLIVE_CAPTURE_ROUTE_REUSE", "0").lower() not in {
            "1",
            "true",
            "yes",
        }:
            return
        requested_layers = {
            int(item)
            for item in os.environ.get(
                "LONGLIVE_CAPTURE_ROUTE_LAYERS", "0,9,19,29"
            ).split(",")
            if item.strip()
        }
        if requested_layers and self.layer_id not in requested_layers:
            return
        marker = (self.layer_id, int(current_start))
        pass_index = self._route_capture_counts.get(marker, 0)
        limit = int(os.environ.get("LONGLIVE_CAPTURE_ROUTE_PASSES", "5"))
        if pass_index >= limit:
            return
        output_root = self._capture_root("route_reuse")
        output_root.mkdir(parents=True, exist_ok=True)
        coordinates = torch.stack(
            (route_plan.union_frame_ids.long(), route_plan.union_token_ids.long()),
            dim=-1,
        )
        capture_kv_hash = os.environ.get(
            "LONGLIVE_CAPTURE_ROUTE_KV_HASH", "0"
        ).lower() in {"1", "true", "yes"}
        torch.save(
            {
                "layer": self.layer_id,
                "current_start": int(current_start),
                "denoising_pass": pass_index,
                "route_plan": route_plan.state_dict(),
                "route_plan_sha256": route_plan.digest(),
                "selected_coordinate_sha256": tensor_sha256(coordinates),
                "transfer_plan_sha256": (
                    transfer_plan.digest() if transfer_plan is not None else None
                ),
                "cache_hit": bool(materialized.cache_hit) if materialized else False,
                "cache_hit_bytes": int(materialized.cache_hit_bytes) if materialized else 0,
                "cache_miss_bytes": int(materialized.cache_miss_bytes) if materialized else 0,
                "key_unrotated_sha256": (
                    tensor_sha256(materialized.key_unrotated)
                    if capture_kv_hash and materialized is not None
                    else None
                ),
                "value_sha256": (
                    tensor_sha256(materialized.value)
                    if capture_kv_hash and materialized is not None
                    else None
                ),
                "rope_position_sha256": (
                    tensor_sha256(materialized.positions)
                    if capture_kv_hash and materialized is not None
                    else None
                ),
                "archive_epoch": self.history_archive.epoch,
                "storage_version": self.history_archive.layer_storage_version(
                    self.layer_id
                ),
            },
            output_root
            / (
                f"layer{self.layer_id:02d}_start{int(current_start):08d}"
                f"_pass{pass_index:02d}.pt"
            ),
        )
        self._route_capture_counts[marker] = pass_index + 1

    def _select_archive(self, query, candidate_frame_ids, current_start):
        if query.shape[0] != 1:
            raise ValueError("LongLive sparse history runtime currently requires batch_size=1")
        global_frame_ids = candidate_frame_ids[0].to(torch.long) + int(self.sink_size)
        cache_key = (
            int(current_start),
            tuple(int(value) for value in global_frame_ids.detach().cpu().tolist()),
        )
        if self.sparse_config.refresh_policy == "per_chunk" and cache_key in self._selection_cache:
            return self._selection_cache[cache_key], global_frame_ids
        selection = self.history_archive.select(
            self.layer_id,
            query,
            global_frame_ids,
        )
        if self.sparse_config.refresh_policy == "per_chunk":
            self._selection_cache[cache_key] = selection
        return selection, global_frame_ids

    def _selection_from_coordinates(self, frame_ids, token_ids, candidate_tokens):
        selected = int((frame_ids >= 0).sum(dim=-1).max()) if frame_ids.numel() else 0
        return SparseSelection(
            frame_ids=frame_ids,
            token_ids=token_ids,
            scores=torch.zeros_like(frame_ids, dtype=torch.float32),
            candidate_history_tokens=int(candidate_tokens),
            selected_history_tokens=selected,
            candidate_units=int(candidate_tokens),
            selected_units=selected,
            cluster_size_min=None,
            cluster_size_max=None,
            index_bytes=0,
            timing=TimingBreakdown(),
        )

    @staticmethod
    def _union_indices_from_coordinates(
        route_plan,
        candidate_frame_ids: torch.Tensor,
        candidate_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map route-plan coordinates into the dense transferred candidate order."""
        return map_union_coordinates(
            route_plan, candidate_frame_ids, candidate_token_ids
        )

    def _materialize_route(
        self,
        route_plan,
        selection: SparseSelection,
        *,
        device: torch.device,
        dtype: torch.dtype,
        current_frame_id: int,
        freqs: torch.Tensor | None,
        candidate_frame_ids: torch.Tensor,
        dense_key: torch.Tensor | None = None,
        dense_value: torch.Tensor | None = None,
        dense_frame_ids: torch.Tensor | None = None,
        dense_token_ids: torch.Tensor | None = None,
    ):
        if isinstance(self.history_union_cache, RawHistoryBlockCache):
            materialized = self.history_archive.materialize_raw_block_cached(
                self.layer_id,
                route_plan,
                self.history_union_cache,
                device=device,
                current_frame_id=current_frame_id,
                freqs=freqs,
                block_tokens=self.sparse_config.block_size,
                candidate_frame_ids=candidate_frame_ids,
            )
            return materialized, None
        candidate_tuple = tuple(
            int(value) for value in candidate_frame_ids.detach().to("cpu").reshape(-1)
        )
        positions = build_sparse_positions(
            frame_ids=route_plan.union_frame_ids.clamp_min(0),
            token_ids=route_plan.union_token_ids.clamp_min(0),
            current_frame_id=current_frame_id,
            spatial_width=self.history_archive.spatial_width,
            rope_policy=self.sparse_config.rope_policy,
            max_relative_age=self.sparse_config.max_relative_age,
            candidate_frame_ids=torch.tensor(candidate_tuple, dtype=torch.long),
        )
        cache_key = None
        if self.history_union_cache is not None:
            self.history_union_cache.begin_chunk(
                current_frame_id,
                per_chunk=self.system_config.gpu_union_cache == "per_chunk",
            )
            coordinates = torch.stack(
                (route_plan.union_frame_ids.long(), route_plan.union_token_ids.long()),
                dim=-1,
            )
            cache_key = HistoryKVCacheKey(
                layer_id=self.layer_id,
                archive_epoch=self.history_archive.epoch,
                storage_version=self.history_archive.layer_storage_version(
                    self.layer_id
                ),
                current_frame_id=current_frame_id,
                candidate_frame_ids=candidate_tuple,
                selected_coordinate_sha256=tensor_sha256(coordinates),
                route_plan_sha256=route_plan.digest(),
                rope_policy=self.sparse_config.rope_policy,
                rope_position_sha256=tensor_sha256(positions),
                dtype=str(dtype),
                device=str(device),
                transfer_layout=self.system_config.transfer_layout,
                padding_strategy="rectangular_head_max",
            )
            cached = self.history_union_cache.get(cache_key)
            if cached is not None:
                rope_start = time.perf_counter()
                key = cached.key_roped
                if key is None:
                    key = cached.key_unrotated
                    if freqs is not None:
                        key = apply_selected_rope(
                            key,
                            cached.positions.to(device),
                            freqs.to(device),
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                rope_s = time.perf_counter() - rope_start
                return (
                    MaterializedHistory(
                        key_unrotated=cached.key_unrotated,
                        key=key,
                        value=cached.value,
                        positions=cached.positions,
                        transferred_bytes=0,
                        cpu_gather_s=0.0,
                        h2d_s=0.0,
                        rope_s=rope_s,
                        transfer_plan_sha256=cached.transfer_plan_sha256,
                        payload_bytes=0,
                        padding_bytes=0,
                        source_run_count=0,
                        cache_hit=True,
                    ),
                    None,
                )

        transfer_plan = None
        dense_arguments = (dense_key, dense_value, dense_frame_ids, dense_token_ids)
        can_use_transfer_plan = (
            self.system_config.transfer_layout != "legacy"
            and not any(value is not None for value in dense_arguments)
        )
        if can_use_transfer_plan:
            bytes_per_token = 2 * self.head_dim * torch.empty((), dtype=dtype).element_size()
            transfer_plan = build_transfer_plan(
                route_plan,
                candidate_tuple,
                frame_tokens=self.history_archive.spatial_height
                * self.history_archive.spatial_width,
                layout=self.system_config.transfer_layout,
                page_tokens=self.system_config.page_tokens,
                bytes_per_token=bytes_per_token,
            )
            materialized = self.history_archive.materialize_transfer_plan(
                self.layer_id,
                transfer_plan,
                route_plan,
                device=device,
                current_frame_id=current_frame_id,
                freqs=freqs,
                staging_pool=self.history_staging_pool,
                staging_mode=self.system_config.staging_mode,
            )
        else:
            materialized = self.history_archive.materialize(
                self.layer_id,
                selection,
                device=device,
                current_frame_id=current_frame_id,
                freqs=freqs,
                candidate_frame_ids=candidate_frame_ids,
                dense_key=dense_key,
                dense_value=dense_value,
                dense_frame_ids=dense_frame_ids,
                dense_token_ids=dense_token_ids,
            )
        if cache_key is not None:
            self.history_union_cache.put(
                CachedHistoryKV(
                    key=cache_key,
                    value=materialized.value.detach(),
                    key_unrotated=materialized.key_unrotated.detach(),
                    key_roped=(
                        materialized.key.detach()
                        if self.system_config.cache_payload == "roped_kv"
                        else None
                    ),
                    positions=materialized.positions.detach(),
                    transfer_plan_sha256=materialized.transfer_plan_sha256,
                )
            )
        return materialized, transfer_plan

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        memory_indices=None,
    ):
        if kv_cache is None or self.sparse_config.method == "dense_history":
            return super().forward(
                x,
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache=kv_cache,
                current_start=current_start,
                cache_start=cache_start,
                sink_recache_after_switch=sink_recache_after_switch,
                memory_indices=memory_indices,
            )

        total_start = time.perf_counter()
        batch, sequence = x.shape[:2]
        heads, dim = self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start
        query = self.norm_q(self.q(x)).view(batch, sequence, heads, dim)
        key = self.norm_k(self.k(x)).view(batch, sequence, heads, dim)
        value = self.v(x).view(batch, sequence, heads, dim)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        num_new_frames = int(grid_sizes[0][0].item())
        current_end = int(current_start + query.shape[1])
        sink_tokens = int(self.sink_size * frame_seqlen)
        kv_cache_size = int(kv_cache["k"].shape[1])
        num_new_tokens = int(query.shape[1])
        cache_update_info = None
        is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0

        if (
            self.local_attn_size != -1
            and current_end > kv_cache["global_end_index"].item()
            and num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size
        ):
            num_evicted_tokens = (
                num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            )
            num_rolled_tokens = (
                kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            )
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
                - num_evicted_tokens
            )
            local_start_index = local_end_index - num_new_tokens
            temp_key = kv_cache["k"].clone()
            temp_value = kv_cache["v"].clone()

            evicted_key_frames = []
            evicted_value_frames = []
            if self.memory_size > 0 and num_evicted_tokens > 0:
                num_evicted_frames = num_evicted_tokens // frame_seqlen
                evicted_key_gpu = temp_key[:, sink_tokens : sink_tokens + num_evicted_tokens]
                evicted_value_gpu = temp_value[:, sink_tokens : sink_tokens + num_evicted_tokens]
                key_splits = evicted_key_gpu.view(
                    batch, num_evicted_frames, frame_seqlen, heads, dim
                ).split(1, dim=1)
                value_splits = evicted_value_gpu.view(
                    batch, num_evicted_frames, frame_seqlen, heads, dim
                ).split(1, dim=1)
                evicted_key_frames = [item.to("cpu", non_blocking=True) for item in key_splits]
                evicted_value_frames = [item.to("cpu", non_blocking=True) for item in value_splits]
                first_archive_slot = len(kv_cache.get("cpu_k_frames", []))
                for offset, (key_gpu, value_gpu, key_cpu, value_cpu) in enumerate(
                    zip(key_splits, value_splits, evicted_key_frames, evicted_value_frames)
                ):
                    global_frame_id = int(self.sink_size + first_archive_slot + offset)
                    self.history_archive.index_frame(
                        self.layer_id,
                        global_frame_id,
                        key_gpu.squeeze(1),
                        value_gpu.squeeze(1),
                        storage_k=key_cpu.squeeze(1),
                        storage_v=value_cpu.squeeze(1),
                    )

            temp_key[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_key[
                :,
                sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens,
            ].clone()
            temp_value[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_value[
                :,
                sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens,
            ].clone()
            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            key_offset = max(0, write_start_index - local_start_index)
            write_length = max(0, local_end_index - write_start_index)
            if write_length > 0:
                temp_key[:, write_start_index:local_end_index] = key[
                    :, key_offset : key_offset + write_length
                ]
                temp_value[:, write_start_index:local_end_index] = value[
                    :, key_offset : key_offset + write_length
                ]
            cache_update_info = {
                "action": "roll_and_insert",
                "sink_tokens": sink_tokens,
                "num_rolled_tokens": num_rolled_tokens,
                "num_evicted_tokens": num_evicted_tokens,
                "local_start_index": local_start_index,
                "local_end_index": local_end_index,
                "write_start_index": write_start_index,
                "write_end_index": local_end_index,
                "new_k": key[:, key_offset : key_offset + write_length],
                "new_v": value[:, key_offset : key_offset + write_length],
                "current_end": current_end,
                "is_recompute": is_recompute,
                "evicted_k_frames": evicted_key_frames,
                "evicted_v_frames": evicted_value_frames,
            }
        else:
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
            )
            local_start_index = local_end_index - num_new_tokens
            temp_key = kv_cache["k"].clone()
            temp_value = kv_cache["v"].clone()
            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            if sink_recache_after_switch:
                write_start_index = local_start_index
            key_offset = max(0, write_start_index - local_start_index)
            write_length = max(0, local_end_index - write_start_index)
            if write_length > 0:
                temp_key[:, write_start_index:local_end_index] = key[
                    :, key_offset : key_offset + write_length
                ]
                temp_value[:, write_start_index:local_end_index] = value[
                    :, key_offset : key_offset + write_length
                ]
            cache_update_info = {
                "action": "direct_insert",
                "local_start_index": local_start_index,
                "local_end_index": local_end_index,
                "write_start_index": write_start_index,
                "write_end_index": local_end_index,
                "new_k": key[:, key_offset : key_offset + write_length],
                "new_v": value[:, key_offset : key_offset + write_length],
                "current_end": current_end,
                "is_recompute": is_recompute,
            }

        query_start_frame = max(0, local_start_index // frame_seqlen)
        query_relative_indices = torch.arange(
            query_start_frame,
            query_start_frame + num_new_frames,
            device=query.device,
        )
        roped_query = causal_online_rope(
            query,
            grid_sizes,
            freqs,
            relative_frame_indices=query_relative_indices,
        ).type_as(value)
        num_cache_frames = local_end_index // frame_seqlen
        cache_grid_sizes = grid_sizes.clone()
        cache_grid_sizes[0, 0] = num_cache_frames
        cache_relative_indices = torch.arange(0, num_cache_frames, device=key.device)
        roped_temp_key = causal_online_rope(
            temp_key[:, :local_end_index]
            .view(batch, num_cache_frames, frame_seqlen, heads, dim)
            .flatten(1, 2),
            cache_grid_sizes,
            freqs,
            relative_frame_indices=cache_relative_indices,
        ).type_as(value)

        backend = attention_backend()
        call_timing = TimingBreakdown()
        if self.memory_size > 0:
            local_budget = self.max_attention_size - sink_tokens - self.memory_size * frame_seqlen
            local_start_for_window = max(sink_tokens, local_end_index - local_budget)
            exact_key_parts = [roped_temp_key[:, :sink_tokens]]
            exact_value_parts = [temp_value[:, :sink_tokens]]
            selected_history_tokens = 0
            candidate_history_tokens = 0
            selected_units = 0
            candidate_units = 0
            cluster_min = None
            cluster_max = None
            transferred_bytes = 0
            candidate_transfer_bytes = 0
            index_bytes = 0
            query_summary_bytes = 0
            selected_transfer_bytes = 0
            if local_budget > 0 and local_start_for_window < local_end_index:
                exact_key_parts.append(roped_temp_key[:, local_start_for_window:local_end_index])
                exact_value_parts.append(temp_value[:, local_start_for_window:local_end_index])
            exact_key = torch.cat(exact_key_parts, dim=1)
            exact_value = torch.cat(exact_value_parts, dim=1)
            exact_tokens = sink_tokens + max(0, local_end_index - local_start_for_window)
            route_plan = None
            transfer_plan = None
            backend_result = None
            materialized = None
            if memory_indices is not None and memory_indices.numel() > 0:
                global_frame_ids = memory_indices[0].to(torch.long) + int(self.sink_size)
                route_cache_key = (
                    "route_plan",
                    self.sparse_config.method,
                    self.sparse_config.history_density,
                    self.sparse_config.rope_policy,
                    tuple(
                        sorted(
                            (name, repr(value))
                            for name, value in self.sparse_config.method_params.items()
                        )
                    ),
                    int(current_start),
                    tuple(int(value) for value in global_frame_ids.detach().cpu().tolist()),
                )
                reuse_route_plan = (
                    self.sparse_config.refresh_policy == "per_chunk"
                    or self.sparse_config.history_density == 1.0
                )
                cached_plan = (
                    self._selection_cache.get(route_cache_key)
                    if reuse_route_plan
                    else None
                )
                spec = method_spec(self.sparse_config.method)
                candidate_history_tokens, candidate_transfer_bytes = (
                    self.history_archive.candidate_history_size(
                        self.layer_id, global_frame_ids
                    )
                )
                candidate_key_cpu = None
                candidate_value_cpu = None
                candidate_frames_cpu = None
                candidate_tokens_cpu = None
                capture_requested = os.environ.get(
                    "LONGLIVE_CAPTURE_QKV", "0"
                ).lower() in {"1", "true", "yes"}

                def ensure_dense_candidate() -> None:
                    nonlocal candidate_key_cpu, candidate_value_cpu
                    nonlocal candidate_frames_cpu, candidate_tokens_cpu
                    if candidate_key_cpu is not None:
                        return
                    (
                        candidate_key_cpu,
                        candidate_value_cpu,
                        candidate_frames_cpu,
                        candidate_tokens_cpu,
                    ) = self.history_archive.dense_history_tensors(
                        self.layer_id, global_frame_ids
                    )
                    self._capture_qkv_once(
                        current_start=current_start,
                        query=query,
                        key=candidate_key_cpu,
                        value=candidate_value_cpu,
                        frame_ids=candidate_frames_cpu,
                        token_ids=candidate_tokens_cpu,
                    )

                if spec.routing_stage != "pre-transfer":
                    ensure_dense_candidate()
                route_start = time.perf_counter()
                if cached_plan is not None:
                    route_plan = cached_plan
                    if route_plan.unique_history_tokens:
                        selection = self._selection_from_coordinates(
                            route_plan.union_frame_ids,
                            route_plan.union_token_ids,
                            candidate_history_tokens,
                        )
                        if spec.routing_stage == "pre-transfer":
                            materialized, transfer_plan = self._materialize_route(
                                route_plan,
                                selection,
                                device=query.device,
                                dtype=query.dtype,
                                current_frame_id=current_start // frame_seqlen,
                                freqs=freqs,
                                candidate_frame_ids=global_frame_ids,
                            )
                        else:
                            ensure_dense_candidate()
                            materialized = self.history_archive.materialize(
                                self.layer_id,
                                selection,
                                device=query.device,
                                current_frame_id=current_start // frame_seqlen,
                                freqs=freqs,
                                candidate_frame_ids=global_frame_ids,
                                dense_key=candidate_key_cpu,
                                dense_value=candidate_value_cpu,
                                dense_frame_ids=candidate_frames_cpu,
                                dense_token_ids=candidate_tokens_cpu,
                            )
                        backend_history_key = materialized.key
                        backend_history_value = materialized.value
                    else:
                        backend_history_key = torch.empty(
                            (batch, 0, heads, dim), dtype=value.dtype, device=query.device
                        )
                        backend_history_value = torch.empty_like(backend_history_key)
                elif spec.routing_stage == "pre-transfer":
                    if self.sparse_config.method in INDEXED_PRETRANSFER_METHODS:
                        if self.sparse_config.method in SUMMARY_PRETRANSFER_METHODS:
                            query_block_size = int(
                                self.sparse_config.method_params.get(
                                    "query_block_size",
                                    spec.query_block_size
                                    or self.sparse_config.block_size,
                                )
                            )
                            route_query = summarize_query_for_pretransfer(
                                query.detach(), query_block_size
                            )
                            call_timing.q_summary_s += route_query.q_summary_s
                            call_timing.d2h_s += route_query.d2h_s
                            query_summary_bytes = route_query.summary_bytes
                        else:
                            route_query = query.detach().to("cpu")
                        route_plan = self.history_archive.route_indexed(
                            self.layer_id,
                            route_query,
                            global_frame_ids,
                            exact_k_tokens=exact_tokens,
                        )
                    else:
                        ensure_dense_candidate()
                        route_plan = route_history(
                            query.detach().to("cpu"),
                            candidate_key_cpu,
                            candidate_frames_cpu,
                            candidate_tokens_cpu,
                            method=self.sparse_config.method,
                            density=self.sparse_config.history_density,
                            exact_k_tokens=exact_tokens,
                            seed=self.sparse_config.seed + self.layer_id * 1009,
                            spec_override=(self.sparse_config.method_params or None),
                        )
                    if route_plan.unique_history_tokens:
                        selection = self._selection_from_coordinates(
                            route_plan.union_frame_ids,
                            route_plan.union_token_ids,
                            candidate_history_tokens,
                        )
                        materialized, transfer_plan = self._materialize_route(
                            route_plan,
                            selection,
                            device=query.device,
                            dtype=query.dtype,
                            current_frame_id=current_start // frame_seqlen,
                            freqs=freqs,
                            candidate_frame_ids=global_frame_ids,
                        )
                        backend_history_key = materialized.key
                        backend_history_value = materialized.value
                    else:
                        backend_history_key = torch.empty(
                            (batch, 0, heads, dim), dtype=value.dtype, device=query.device
                        )
                        backend_history_value = torch.empty_like(backend_history_key)
                else:
                    dense_selection = self._selection_from_coordinates(
                        candidate_frames_cpu,
                        candidate_tokens_cpu,
                        candidate_history_tokens,
                    )
                    materialized = self.history_archive.materialize(
                        self.layer_id,
                        dense_selection,
                        device=query.device,
                        current_frame_id=current_start // frame_seqlen,
                        freqs=freqs,
                        candidate_frame_ids=global_frame_ids,
                        dense_key=candidate_key_cpu,
                        dense_value=candidate_value_cpu,
                        dense_frame_ids=candidate_frames_cpu,
                        dense_token_ids=candidate_tokens_cpu,
                    )
                    route_query = roped_query if self.sparse_config.method == "scope_ar" else query
                    route_key = materialized.key if self.sparse_config.method == "scope_ar" else materialized.key_unrotated
                    route_plan = route_history(
                        route_query,
                        route_key,
                        candidate_frames_cpu.to(query.device),
                        candidate_tokens_cpu.to(query.device),
                        method=self.sparse_config.method,
                        density=self.sparse_config.history_density,
                        exact_k_tokens=exact_tokens,
                        seed=self.sparse_config.seed + self.layer_id * 1009,
                        spec_override=(self.sparse_config.method_params or None),
                    )
                    union_indices = self._union_indices_from_coordinates(
                        route_plan,
                        candidate_frames_cpu,
                        candidate_tokens_cpu,
                    )
                    backend_history_key = gather_per_head(
                        materialized.key, union_indices
                    )
                    backend_history_value = gather_per_head(
                        materialized.value, union_indices
                    )
                if reuse_route_plan:
                    self._selection_cache[route_cache_key] = route_plan
                materialization_s = (
                    materialized.cpu_gather_s
                    + materialized.h2d_s
                    + materialized.rope_s
                    if materialized is not None
                    else 0.0
                )
                call_timing.routing_s = max(
                    0.0,
                    time.perf_counter()
                    - route_start
                    - call_timing.q_summary_s
                    - call_timing.d2h_s
                    - materialization_s,
                )
                if spec.routing_stage == "pre-transfer" and capture_requested:
                    ensure_dense_candidate()
                index_bytes = int(route_plan.metadata.get("routing_index_bytes", 0))
                backend_result = execute_plan(
                    self.sparse_config.backend,
                    roped_query,
                    exact_key,
                    exact_value,
                    backend_history_key,
                    backend_history_value,
                    route_plan,
                )
                output = backend_result.output
                call_timing.attention_s = backend_result.elapsed_ms / 1000.0
                selected_history_tokens = route_plan.unique_history_tokens
                selected_units = route_plan.groups
                candidate_units = candidate_history_tokens
                transferred_bytes = materialized.transferred_bytes if materialized else 0
                selected_transfer_bytes = (
                    route_plan.unique_history_tokens
                    * 2
                    * dim
                    * value.element_size()
                )
                staging_padding_tokens = (
                    backend_history_key.shape[0]
                    * backend_history_key.shape[1]
                    * backend_history_key.shape[2]
                    - route_plan.unique_history_tokens
                )
                if materialized is not None:
                    call_timing.cpu_gather_s = materialized.cpu_gather_s
                    call_timing.h2d_s = materialized.h2d_s
                    call_timing.rope_s = materialized.rope_s
                self._capture_route_reuse(
                    current_start=current_start,
                    route_plan=route_plan,
                    transfer_plan=transfer_plan,
                    materialized=materialized,
                )
            else:
                output, call_timing.attention_s = _timed_attention(
                    roped_query, exact_key, exact_value
                )
            dense_key_tokens = exact_tokens + candidate_history_tokens
            record = SparseCallRecord(
                layer_id=self.layer_id,
                method=self.sparse_config.method,
                candidate_frames=(
                    int(memory_indices.shape[1]) if memory_indices is not None else 0
                ),
                candidate_history_tokens=candidate_history_tokens,
                selected_history_tokens=selected_history_tokens,
                exact_tokens=exact_tokens,
                query_tokens=query.shape[1],
                dense_k_tokens=dense_key_tokens,
                executed_k_tokens=(
                    exact_tokens + selected_history_tokens
                    if route_plan is not None
                    else exact_tokens
                ),
                transferred_bytes=transferred_bytes,
                index_bytes=index_bytes,
                query_summary_bytes=query_summary_bytes,
                candidate_transfer_bytes=candidate_transfer_bytes,
                full_history_pairs=(route_plan.full_history_pairs if route_plan else 0),
                selected_history_pairs=(route_plan.history_pairs if route_plan else 0),
                dense_qk_pairs_value=(
                    query.shape[0] * query.shape[2] * query.shape[1] * exact_tokens
                    + (route_plan.full_history_pairs if route_plan else 0)
                ),
                executed_qk_pairs_value=(
                    query.shape[0] * query.shape[2] * query.shape[1] * exact_tokens
                    + (route_plan.history_pairs if route_plan else 0)
                ),
                cluster_size_min=cluster_min,
                cluster_size_max=cluster_max,
                selected_units=selected_units,
                candidate_units=candidate_units,
                attention_backend=(backend_result.backend if backend_result else backend),
                routing_stage=self.sparse_config.routing_stage,
                history_pair_density_value=(route_plan.history_pair_density if route_plan else 0.0),
                history_transfer_density=(
                    transferred_bytes / candidate_transfer_bytes
                    if candidate_transfer_bytes
                    else None
                ),
                staging_padding_tokens=(staging_padding_tokens if route_plan else 0),
                scheduled_pairs=(backend_result.scheduled_pairs if backend_result else query.shape[1] * exact_tokens),
                route_plan_sha256=(route_plan.digest() if route_plan else None),
                transfer_plan_sha256=(
                    materialized.transfer_plan_sha256
                    if materialized is not None
                    else None
                ),
                transfer_layout=(
                    self.system_config.transfer_layout
                    if route_plan is not None
                    else "legacy"
                ),
                transfer_payload_bytes=(
                    int(
                        materialized.payload_bytes
                        if materialized is not None
                        and materialized.payload_bytes is not None
                        else (
                            selected_transfer_bytes
                            if materialized is not None and not materialized.cache_hit
                            else 0
                        )
                    )
                ),
                transfer_padding_bytes=(
                    materialized.padding_bytes if materialized is not None else 0
                ),
                transfer_source_runs=(
                    materialized.source_run_count if materialized is not None else 0
                ),
                cache_hit_bytes=(
                    materialized.cache_hit_bytes
                    if materialized is not None and materialized.cache_hit_bytes
                    else (
                        selected_transfer_bytes
                        if materialized is not None and materialized.cache_hit
                        else 0
                    )
                ),
                cache_miss_bytes=(
                    materialized.cache_miss_bytes
                    if materialized is not None and materialized.cache_miss_bytes
                    else (
                        selected_transfer_bytes
                        if materialized is not None and not materialized.cache_hit
                        else 0
                    )
                ),
                h2d_copy_count=(
                    materialized.h2d_copy_count if materialized is not None else 0
                ),
                staging_reuse_count=(
                    1
                    if materialized is not None and materialized.staging_reused
                    else 0
                ),
                timing=call_timing,
            )
        else:
            if self.sparse_config.method not in {"block64_history", "native_block"}:
                raise ValueError(
                    "native rolling-cache mode currently supports block64_history only"
                )
            local_start_for_window = max(sink_tokens, local_end_index - self.max_attention_size + sink_tokens)
            recent_tokens = min(
                self.sparse_config.recent_exact_frames * frame_seqlen,
                max(0, local_end_index - local_start_for_window),
            )
            old_start = local_start_for_window
            old_end = max(old_start, local_end_index - recent_tokens)
            key_parts = [roped_temp_key[:, :sink_tokens]]
            value_parts = [temp_value[:, :sink_tokens]]
            selected_history_tokens = 0
            route_start = time.perf_counter()
            if old_end > old_start:
                selected_indices = select_block64_from_tensor(
                    query,
                    temp_key[:, old_start:old_end],
                    self.sparse_config.history_density,
                    self.sparse_config.block_size,
                )
                key_parts.append(
                    gather_per_head(roped_temp_key[:, old_start:old_end], selected_indices)
                )
                value_parts.append(
                    gather_per_head(temp_value[:, old_start:old_end], selected_indices)
                )
                selected_history_tokens = selected_indices.shape[-1]
            call_timing.routing_s = time.perf_counter() - route_start
            key_parts.append(roped_temp_key[:, old_end:local_end_index])
            value_parts.append(temp_value[:, old_end:local_end_index])
            key_cat = torch.cat(key_parts, dim=1)
            value_cat = torch.cat(value_parts, dim=1)
            output, call_timing.attention_s = _timed_attention(
                roped_query, key_cat, value_cat
            )
            candidate_history_tokens = max(0, old_end - old_start)
            exact_tokens = sink_tokens + max(0, local_end_index - old_end)
            record = SparseCallRecord(
                layer_id=self.layer_id,
                method=self.sparse_config.method,
                candidate_frames=candidate_history_tokens // frame_seqlen,
                candidate_history_tokens=candidate_history_tokens,
                selected_history_tokens=selected_history_tokens,
                exact_tokens=exact_tokens,
                query_tokens=query.shape[1],
                dense_k_tokens=exact_tokens + candidate_history_tokens,
                executed_k_tokens=key_cat.shape[1],
                transferred_bytes=0,
                index_bytes=0,
                full_history_pairs=(
                    query.shape[0]
                    * query.shape[2]
                    * query.shape[1]
                    * candidate_history_tokens
                ),
                selected_history_pairs=(
                    query.shape[0]
                    * query.shape[2]
                    * query.shape[1]
                    * selected_history_tokens
                ),
                dense_qk_pairs_value=(
                    query.shape[0]
                    * query.shape[2]
                    * query.shape[1]
                    * (exact_tokens + candidate_history_tokens)
                ),
                executed_qk_pairs_value=(
                    query.shape[0]
                    * query.shape[2]
                    * query.shape[1]
                    * key_cat.shape[1]
                ),
                candidate_units=math.ceil(candidate_history_tokens / self.sparse_config.block_size),
                selected_units=math.ceil(selected_history_tokens / self.sparse_config.block_size),
                attention_backend=backend,
                routing_stage=self.sparse_config.routing_stage,
                timing=call_timing,
            )

        call_timing.total_s = time.perf_counter() - total_start
        self.history_archive.stats.record_call(
            record,
            keep_detail=self.sparse_config.record_per_call,
        )
        output = self.o(output.flatten(2))
        return output, (current_end, local_end_index, cache_update_info)


def install_sparse_history_attention(
    model,
    archive: HistoryArchive,
    config: SparseHistoryConfig,
    *,
    system_config: LongLiveSystemConfig | None = None,
    history_union_cache: HistoryUnionCache | RawHistoryBlockCache | None = None,
    history_staging_pool: PinnedStagingPool | None = None,
) -> list[SparseHistorySelfAttention]:
    """Replace all LongLive-RAG self-attention modules without changing weights."""

    installed = []
    for layer_id, block in enumerate(model.blocks):
        original = block.self_attn
        replacement = SparseHistorySelfAttention(
            dim=original.dim,
            num_heads=original.num_heads,
            local_attn_size=original.local_attn_size,
            sink_size=original.sink_size,
            memory_size=original.memory_size,
            qk_norm=original.qk_norm,
            eps=original.eps,
            layer_id=layer_id,
            history_archive=archive,
            sparse_config=config,
            system_config=system_config,
            history_union_cache=history_union_cache,
            history_staging_pool=history_staging_pool,
        )
        replacement.load_state_dict(original.state_dict(), strict=True)
        parameter = next(original.parameters())
        replacement.to(device=parameter.device, dtype=parameter.dtype)
        replacement.train(original.training)
        block.self_attn = replacement
        installed.append(replacement)
    if not installed:
        raise RuntimeError("no LongLive self-attention modules were replaced")
    return installed
