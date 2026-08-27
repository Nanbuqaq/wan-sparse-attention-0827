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

from .archive import HistoryArchive
from .ar_routing import route_history
from .backends import execute_plan
from .config import SparseHistoryConfig
from .methods import method_spec
from .selectors import SparseSelection, gather_per_head, select_block64_from_tensor
from .selectors import INDEXED_PRETRANSFER_METHODS
from .stats import SparseCallRecord, TimingBreakdown
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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layer_id = int(layer_id)
        self.history_archive = history_archive
        self.sparse_config = sparse_config
        self._selection_cache: dict[tuple[Any, ...], Any] = {}
        self._captured_qkv: set[tuple[int, int]] = set()
        self._capture_counts: dict[int, int] = {}

    def clear_selection_cache(self) -> None:
        self._selection_cache.clear()

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
        marker = (self.layer_id, int(current_start))
        if self.layer_id not in layers or marker in self._captured_qkv:
            return
        max_per_layer = int(os.environ.get("LONGLIVE_CAPTURE_MAX_PER_LAYER", "0"))
        if max_per_layer > 0 and self._capture_counts.get(self.layer_id, 0) >= max_per_layer:
            return
        output_root = Path(os.environ.get("INFER_OUTPUT_DIR", "results/captures")) / "qkv_captures"
        output_root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layer": self.layer_id,
                "current_start": int(current_start),
                "query": query.detach().to("cpu"),
                "key": key.detach().to("cpu"),
                "value": value.detach().to("cpu"),
                "frame_ids": frame_ids.detach().to("cpu"),
                "token_ids": token_ids.detach().to("cpu"),
            },
            output_root / f"layer{self.layer_id:02d}_start{int(current_start):08d}.pt",
        )
        self._captured_qkv.add(marker)
        self._capture_counts[self.layer_id] = self._capture_counts.get(self.layer_id, 0) + 1

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

        frame_ids = route_plan.union_frame_ids
        token_ids = route_plan.union_token_ids
        valid = frame_ids >= 0
        max_token = max(
            int(candidate_token_ids.max()) if candidate_token_ids.numel() else 0,
            int(token_ids[valid].max()) if valid.any() else 0,
        )
        base = max_token + 1
        candidate_codes = candidate_frame_ids.long() * base + candidate_token_ids.long()
        union_codes = frame_ids.long() * base + token_ids.clamp_min(0).long()
        sorted_codes, sorted_to_dense = torch.sort(candidate_codes, dim=-1)
        sorted_indices = torch.searchsorted(
            sorted_codes.contiguous(), union_codes.contiguous()
        ).clamp_max(candidate_codes.shape[-1] - 1)
        indices = sorted_to_dense.gather(-1, sorted_indices)
        matched = candidate_codes.gather(-1, indices) == union_codes
        if not bool((matched | ~valid).all()):
            raise KeyError("route plan contains coordinates outside the dense candidate transfer")
        return torch.where(valid, indices, torch.zeros_like(indices))

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
            if local_budget > 0 and local_start_for_window < local_end_index:
                exact_key_parts.append(roped_temp_key[:, local_start_for_window:local_end_index])
                exact_value_parts.append(temp_value[:, local_start_for_window:local_end_index])
            exact_key = torch.cat(exact_key_parts, dim=1)
            exact_value = torch.cat(exact_value_parts, dim=1)
            exact_tokens = sink_tokens + max(0, local_end_index - local_start_for_window)
            route_plan = None
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
                candidate_key_cpu, candidate_value_cpu, candidate_frames_cpu, candidate_tokens_cpu = self.history_archive.dense_history_tensors(
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
                candidate_history_tokens = candidate_key_cpu.shape[1]
                candidate_transfer_bytes = (
                    candidate_key_cpu.numel() + candidate_value_cpu.numel()
                ) * candidate_key_cpu.element_size()
                spec = method_spec(self.sparse_config.method)
                route_start = time.perf_counter()
                if cached_plan is not None:
                    route_plan = cached_plan
                    if route_plan.unique_history_tokens:
                        selection = self._selection_from_coordinates(
                            route_plan.union_frame_ids,
                            route_plan.union_token_ids,
                            candidate_history_tokens,
                        )
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
                        backend_history_key = candidate_key_cpu[:, :0].to(query.device)
                        backend_history_value = candidate_value_cpu[:, :0].to(query.device)
                elif spec.routing_stage == "pre-transfer":
                    if self.sparse_config.method in INDEXED_PRETRANSFER_METHODS:
                        route_plan = self.history_archive.route_indexed(
                            self.layer_id,
                            query.detach().to("cpu"),
                            global_frame_ids,
                            exact_k_tokens=exact_tokens,
                        )
                    else:
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
                        backend_history_key = candidate_key_cpu[:, :0].to(query.device)
                        backend_history_value = candidate_value_cpu[:, :0].to(query.device)
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
                call_timing.routing_s = time.perf_counter() - route_start
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
