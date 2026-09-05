"""CPU history archive, exact token materialization, and transfer accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import SparseHistoryConfig
from .contexts import OnlineRoutingContext
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
from .history_cache import (
    CachedRawHistoryBlock,
    RawHistoryBlockCache,
    RawHistoryBlockCacheKey,
)
from .staging import PinnedStagingPool
from .transfer_plan import TransferPlan
from .archive_pack import pack_archive_runs
from .profiling import profiled


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
    transfer_plan_sha256: str | None = None
    payload_bytes: int | None = None
    padding_bytes: int = 0
    source_run_count: int = 0
    cache_hit: bool = False
    h2d_copy_count: int = 0
    staging_mode: str = "per_call_separate"
    staging_reused: bool = False
    cache_hit_bytes: int = 0
    cache_miss_bytes: int = 0
    cpu_prepare_s: float = 0.0
    cpu_pack_s: float = 0.0
    cpu_allocate_pin_s: float = 0.0
    gpu_restore_s: float = 0.0
    materialize_total_s: float = 0.0
    h2d_device_s: float | None = None


class HistoryArchive:
    """Per-layer archive of original unrotated K/V and retrieval metadata."""

    def __init__(self, config: SparseHistoryConfig, *, spatial_height: int, spatial_width: int):
        if spatial_height < 1 or spatial_width < 1:
            raise ValueError("spatial dimensions must be positive")
        self.config = config
        self.spatial_height = int(spatial_height)
        self.spatial_width = int(spatial_width)
        self._layers: dict[int, dict[int, FrameIndex]] = {}
        self._epoch = 0
        self._storage_version = 0
        self._layer_storage_versions: dict[int, int] = {}
        self._frame_storage_versions: dict[tuple[int, int], int] = {}
        self.stats = SparseRunStats(method=config.method)

    def reset(self) -> None:
        self.clear_frames()
        self.stats = SparseRunStats(method=self.config.method)

    def clear_frames(self) -> None:
        self._layers.clear()
        self._epoch += 1
        self._storage_version = 0
        self._layer_storage_versions.clear()
        self._frame_storage_versions.clear()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def storage_version(self) -> int:
        return self._storage_version

    def layer_storage_version(self, layer_id: int) -> int:
        return self._layer_storage_versions.get(int(layer_id), 0)

    def frame_ids(self, layer_id: int) -> list[int]:
        return sorted(self._layers.get(int(layer_id), {}))

    def frame_storage_version(self, layer_id: int, frame_id: int) -> int:
        try:
            return self._frame_storage_versions[(int(layer_id), int(frame_id))]
        except KeyError as error:
            raise KeyError(
                f"frame {frame_id} is not indexed for layer {layer_id}"
            ) from error

    @profiled("history/archive_offload_and_index")
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
        self._storage_version += 1
        self._frame_storage_versions[(layer_id, frame_id)] = self._storage_version
        self._layer_storage_versions[layer_id] = (
            self._layer_storage_versions.get(layer_id, 0) + 1
        )
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

    def full_history_route(self, layer_id: int, candidate_frame_ids, *,
                           query_shape, exact_k_tokens: int):
        """Dense logical route from coordinates alone, without assembling raw K/V."""
        from .ar_routing import build_route_plan
        ids = tuple(int(frame) for frame in candidate_frame_ids)
        tokens, _ = self.candidate_history_size(layer_id, ids)
        batch, queries, heads, _ = query_shape
        width = self.spatial_height * self.spatial_width
        frame_ids = torch.tensor(ids).repeat_interleave(width).view(1, 1, -1).expand(batch, heads, -1)
        token_ids = torch.arange(width).repeat(len(ids)).view(1, 1, -1).expand(batch, heads, -1)
        full = torch.arange(tokens)
        return build_route_plan(method=self.config.method, routing_stage=self.config.routing_stage,
            query_labels=torch.zeros(batch, heads, queries, dtype=torch.long),
            selections=[[[full] for _ in range(heads)] for _ in range(batch)],
            history_frame_ids=frame_ids, history_token_ids=token_ids,
            candidate_history_tokens=tokens, exact_k_tokens=exact_k_tokens, density=1.0,
            metadata={'full_density_metadata_only': True})

    @profiled("history/cpu_route_indexed")
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

    def route_system_utility(
        self,
        layer_id: int,
        summary: PretransferQuerySummary,
        candidate_frame_ids: torch.Tensor | list[int],
        *,
        exact_k_tokens: int,
        group_selection_policy: str,
        group_top_p: float,
        group_min_k_ratio: float,
    ):
        """Build the capture-screened online utility route without teacher data."""

        if self.config.method != "system_utility_history":
            raise ValueError("route_system_utility requires system_utility_history config")
        params = dict(self.config.method_params)
        from .methods import method_spec

        spec = method_spec(self.config.method)
        cost_strategy = str(
            params.get("cost_strategy", spec.cost_strategy or "static_block")
        )
        if cost_strategy != "static_block":
            raise ValueError(
                "marginal_set runtime is disabled because held-out cost MAPE exceeded 15%"
            )
        from .system_utility_route import (
            SystemUtilityRouteConfig,
            build_system_utility_route,
        )

        context = self.online_routing_context(
            layer_id, summary, candidate_frame_ids
        )
        route_config = SystemUtilityRouteConfig(
            value_candidate=str(
                params.get("value_candidate", spec.value_candidate or "peak_value")
            ),
            cost_strategy=cost_strategy,
            history_density=self.config.history_density,
            correlation_fraction=float(
                params.get("correlation_fraction", spec.correlation_fraction or 0.70)
            ),
            coverage_fraction=float(
                params.get("coverage_fraction", spec.coverage_fraction or 0.15)
            ),
            remote_fraction=float(
                params.get("remote_fraction", spec.remote_fraction or 0.15)
            ),
            exploration_fraction=float(
                params.get(
                    "exploration_fraction", spec.exploration_fraction or 0.0
                )
            ),
            remote_min_age=int(
                params.get("remote_min_age", spec.remote_min_age or 2)
            ),
            group_selection_policy=group_selection_policy,
            group_top_p=group_top_p,
            group_min_k_ratio=group_min_k_ratio,
        )
        return build_system_utility_route(
            context,
            exact_k_tokens=exact_k_tokens,
            config=route_config,
        )

    def online_routing_context(
        self,
        layer_id: int,
        summary: PretransferQuerySummary,
        candidate_frame_ids: torch.Tensor | list[int],
        *,
        past_attention_score: torch.Tensor | None = None,
        query_role_probabilities: torch.Tensor | None = None,
        block_role_probabilities: torch.Tensor | None = None,
        resident_blocks: torch.Tensor | None = None,
        hardware_profile_id: str | None = None,
        cost_model_version: str | None = None,
    ) -> OnlineRoutingContext:
        """Expose compact online-legal Block64 metadata for co-design routes."""

        if isinstance(candidate_frame_ids, torch.Tensor):
            ids = [
                int(value)
                for value in candidate_frame_ids.detach().to("cpu").reshape(-1)
            ]
        else:
            ids = [int(value) for value in candidate_frame_ids]
        layer = self._layers.get(int(layer_id), {})
        missing = [frame_id for frame_id in ids if frame_id not in layer]
        if missing:
            raise KeyError(f"unindexed history frames for layer {layer_id}: {missing}")
        frames = [layer[frame_id] for frame_id in ids]
        key_prototypes = torch.cat(
            [frame.block_centroids.detach().to("cpu") for frame in frames], dim=2
        )
        value_prototypes = torch.cat(
            [frame.block_value_centroids.detach().to("cpu") for frame in frames], dim=2
        )
        block_frame_ids = []
        block_starts = []
        block_ends = []
        for frame in frames:
            blocks = int(frame.block_starts.numel())
            block_frame_ids.extend([frame.frame_id] * blocks)
            block_starts.extend(int(value) for value in frame.block_starts)
            block_ends.extend(int(value) for value in frame.block_ends)
        newest = max(block_frame_ids)
        return OnlineRoutingContext(
            query_centroids=summary.query_centroids.detach().to("cpu"),
            query_group_sizes=summary.query_group_sizes.detach().to("cpu"),
            key_prototypes=key_prototypes,
            value_prototypes=value_prototypes,
            block_frame_ids=torch.tensor(block_frame_ids, dtype=torch.long),
            block_token_starts=torch.tensor(block_starts, dtype=torch.long),
            block_token_ends=torch.tensor(block_ends, dtype=torch.long),
            block_age=torch.tensor(
                [newest - frame_id for frame_id in block_frame_ids],
                dtype=torch.float32,
            ),
            past_attention_score=past_attention_score,
            query_role_probabilities=query_role_probabilities,
            block_role_probabilities=block_role_probabilities,
            resident_blocks=resident_blocks,
            hardware_profile_id=hardware_profile_id,
            cost_model_version=cost_model_version,
            metadata={
                "layer_id": int(layer_id),
                "candidate_frame_ids": ids,
                "index_source": "per_frame_cpu_block64_kv_prototypes",
                "raw_candidate_kv_exposed": False,
                "archive_dtype": str(frames[0].key.dtype),
                "archive_head_dim": int(frames[0].key.shape[-1]),
                "bytes_per_history_token": int(
                    2
                    * frames[0].key.shape[-1]
                    * frames[0].key.element_size()
                ),
            },
        )

    @profiled("history/legacy_materialize_complete")
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

    @profiled("history/transfer_materialize_complete")
    def materialize_transfer_plan(
        self,
        layer_id: int,
        transfer_plan: TransferPlan,
        route_plan,
        *,
        device: torch.device | str,
        current_frame_id: int,
        freqs: torch.Tensor | None,
        staging_pool: PinnedStagingPool | None = None,
        staging_mode: str = "per_call_separate",
        cpu_pack_policy: str = "candidate_gather",
    ) -> MaterializedHistory:
        """Materialize a physical plan, then expose the original logical union.

        This initial implementation intentionally keeps the existing pageable
        CPU archive as the source of truth.  It makes layout/padding and
        transferred bytes explicit while preserving the exact route output.
        Persistent staging and direct multi-run copies are later optimizations
        behind the same contract.
        """

        total_start = time.perf_counter()
        if transfer_plan.route_plan_sha256 != route_plan.digest():
            raise ValueError("transfer plan does not match route plan")
        if bool(transfer_plan.resident_logical_mask.any()):
            raise NotImplementedError(
                "partial-resident materialization requires cache composition"
            )
        target_device = torch.device(device)
        use_pinned = bool(
            self.config.pin_memory
            and target_device.type == "cuda"
            and torch.cuda.is_available()
        )
        gather_start = time.perf_counter()
        use_pool = staging_pool is not None
        head_major = cpu_pack_policy == "archive_runs"
        if head_major:
            packed = pack_archive_runs(
                self._layers[int(layer_id)], transfer_plan, pin_memory=use_pinned,
                pool=staging_pool, fused=staging_mode == 'persistent_fused',
            )
            key_cpu, value_cpu, lease = packed.key, packed.value, packed.lease
            cpu_prepare_s = packed.cpu_prepare_s
            cpu_pack_s = packed.cpu_pack_s
            cpu_allocate_pin_s = packed.cpu_allocate_pin_s
        elif cpu_pack_policy == 'candidate_gather':
            candidate_key, candidate_value, _, _ = self.dense_history_tensors(
                layer_id, transfer_plan.candidate_frame_ids
            )
            cpu_prepare_s = time.perf_counter() - gather_start
            pack_start = time.perf_counter()
            source_indices = transfer_plan.physical_source_offsets.clamp_min(0)
            physical_key = gather_per_head(candidate_key, source_indices)
            physical_value = gather_per_head(candidate_value, source_indices)
            if source_indices.shape[-1]:
                valid_physical = (
                    torch.arange(source_indices.shape[-1]).view(1, 1, -1)
                    < transfer_plan.physical_counts.unsqueeze(-1)
                )
                physical_mask = valid_physical.permute(0, 2, 1).unsqueeze(-1)
                physical_key = physical_key.masked_fill(~physical_mask, 0)
                physical_value = physical_value.masked_fill(~physical_mask, 0)
            cpu_pack_s = time.perf_counter() - pack_start
            pin_start = time.perf_counter()
            lease = None
            if use_pool:
                fused = staging_mode == "persistent_fused"
                lease = staging_pool.acquire(
                    tuple(physical_key.shape), physical_key.dtype, fused=fused
                )
                key_cpu, value_cpu = lease.key, lease.value
                key_cpu.copy_(physical_key)
                value_cpu.copy_(physical_value)
            elif use_pinned:
                key_cpu = torch.empty_like(physical_key, pin_memory=True)
                value_cpu = torch.empty_like(physical_value, pin_memory=True)
                key_cpu.copy_(physical_key)
                value_cpu.copy_(physical_value)
            else:
                key_cpu, value_cpu = physical_key, physical_value
            cpu_allocate_pin_s = time.perf_counter() - pin_start
        else:
            raise ValueError(f'unsupported cpu_pack_policy: {cpu_pack_policy}')
        cpu_gather_s = time.perf_counter() - gather_start

        transfer_start = time.perf_counter()
        h2d_start = h2d_end = None
        if target_device.type == 'cuda':
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
        if lease is not None and lease.fused is not None:
            fused_device = lease.fused.to(
                target_device,
                non_blocking=use_pinned and self.config.non_blocking_h2d,
            )
            physical_key_device, physical_value_device = fused_device[0], fused_device[1]
            h2d_copy_count = 1
        else:
            physical_key_device = key_cpu.to(
                target_device,
                non_blocking=use_pinned and self.config.non_blocking_h2d,
            )
            physical_value_device = value_cpu.to(
                target_device,
                non_blocking=use_pinned and self.config.non_blocking_h2d,
            )
            h2d_copy_count = 2
        if target_device.type == "cuda":
            h2d_end.record()
            torch.cuda.synchronize(target_device)
        h2d_s = time.perf_counter() - transfer_start
        if lease is not None:
            staging_pool.release(lease)

        restore_start = time.perf_counter()
        if head_major:
            physical_key_device = physical_key_device.permute(0, 2, 1, 3)
            physical_value_device = physical_value_device.permute(0, 2, 1, 3)
        logical_indices = transfer_plan.logical_to_physical.clamp_min(0).to(
            target_device
        )
        key_unrotated = gather_per_head(physical_key_device, logical_indices)
        value = gather_per_head(physical_value_device, logical_indices)
        logical_valid = (route_plan.union_frame_ids >= 0).to(target_device)
        logical_mask = logical_valid.permute(0, 2, 1).unsqueeze(-1)
        key_unrotated = key_unrotated.masked_fill(~logical_mask, 0)
        value = value.masked_fill(~logical_mask, 0)
        if target_device.type == 'cuda':
            torch.cuda.synchronize(target_device)
        gpu_restore_s = time.perf_counter() - restore_start

        positions = build_sparse_positions(
            frame_ids=route_plan.union_frame_ids.clamp_min(0),
            token_ids=route_plan.union_token_ids.clamp_min(0),
            current_frame_id=current_frame_id,
            spatial_width=self.spatial_width,
            rope_policy=self.config.rope_policy,
            max_relative_age=self.config.max_relative_age,
            candidate_frame_ids=torch.tensor(
                transfer_plan.candidate_frame_ids, dtype=torch.long
            ),
        )
        rope_start = time.perf_counter()
        key = key_unrotated
        if freqs is not None:
            key = apply_selected_rope(
                key_unrotated,
                positions.to(target_device),
                freqs.to(target_device),
            )
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
        rope_s = time.perf_counter() - rope_start
        transferred_bytes = (
            physical_key_device.numel() + physical_value_device.numel()
        ) * physical_key_device.element_size()
        payload_bytes = transfer_plan.missing_logical_tokens * transfer_plan.bytes_per_token
        plan_sha = transfer_plan.digest()
        return MaterializedHistory(
            key_unrotated=key_unrotated,
            key=key,
            value=value,
            positions=positions,
            transferred_bytes=transferred_bytes,
            cpu_gather_s=cpu_gather_s,
            h2d_s=h2d_s,
            rope_s=rope_s,
            transfer_plan_sha256=plan_sha,
            payload_bytes=payload_bytes,
            padding_bytes=max(0, transferred_bytes - payload_bytes),
            source_run_count=transfer_plan.source_run_count,
            h2d_copy_count=h2d_copy_count,
            staging_mode=staging_mode,
            staging_reused=(lease.reused if lease is not None else False),
            cpu_prepare_s=cpu_prepare_s,
            cpu_pack_s=cpu_pack_s,
            cpu_allocate_pin_s=cpu_allocate_pin_s,
            gpu_restore_s=gpu_restore_s,
            materialize_total_s=time.perf_counter() - total_start,
            h2d_device_s=(h2d_start.elapsed_time(h2d_end) / 1000 if h2d_start else None),
        )

    def materialize_raw_block_cached(
        self,
        layer_id: int,
        route_plan,
        cache: RawHistoryBlockCache,
        *,
        device: torch.device | str,
        current_frame_id: int,
        freqs: torch.Tensor | None,
        block_tokens: int = 64,
        candidate_frame_ids: torch.Tensor | list[int] | None = None,
    ) -> MaterializedHistory:
        """Compose one logical union from reusable raw Block64 cache entries."""

        if block_tokens < 1:
            raise ValueError("block_tokens must be positive")
        target_device = torch.device(device)
        layer = self._layers.get(int(layer_id), {})
        if not layer:
            raise KeyError(f"layer {layer_id} has no archived frames")
        frames = route_plan.union_frame_ids.detach().to("cpu")
        tokens = route_plan.union_token_ids.detach().to("cpu")
        batch, heads, union_width = frames.shape
        first = next(iter(layer.values()))
        dim = first.key.shape[-1]
        dtype = first.key.dtype
        key_unrotated = torch.zeros(
            (batch, union_width, heads, dim), dtype=dtype, device=target_device
        )
        value = torch.zeros_like(key_unrotated)
        requests: dict[
            tuple[int, int, int, int, int], list[tuple[int, int]]
        ] = {}
        for batch_index in range(batch):
            for head_index in range(heads):
                for union_index in range(union_width):
                    frame_id = int(frames[batch_index, head_index, union_index])
                    token_id = int(tokens[batch_index, head_index, union_index])
                    if frame_id < 0:
                        continue
                    if frame_id not in layer:
                        raise KeyError(
                            f"route frame {frame_id} is outside archived layer {layer_id}"
                        )
                    token_start = token_id // block_tokens * block_tokens
                    token_end = min(
                        layer[frame_id].key.shape[1], token_start + block_tokens
                    )
                    requests.setdefault(
                        (batch_index, head_index, frame_id, token_start, token_end), []
                    ).append((union_index, token_id - token_start))

        gather_start = time.perf_counter()
        missing: list[
            tuple[
                RawHistoryBlockCacheKey,
                torch.Tensor,
                torch.Tensor,
                list[tuple[int, int]],
                int,
            ]
        ] = []
        resident: list[
            tuple[CachedRawHistoryBlock, list[tuple[int, int]], int]
        ] = []
        for (batch_index, head_index, frame_id, token_start, token_end), uses in requests.items():
            cache_key = RawHistoryBlockCacheKey(
                batch_id=batch_index,
                layer_id=int(layer_id),
                head_id=head_index,
                archive_epoch=self.epoch,
                frame_id=frame_id,
                frame_storage_version=self.frame_storage_version(
                    layer_id, frame_id
                ),
                token_start=token_start,
                token_end=token_end,
                dtype=str(dtype),
                device=str(target_device),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                resident.append((cached, uses, batch_index))
                continue
            frame = layer[frame_id]
            source_key = frame.key[
                batch_index, token_start:token_end, head_index
            ].contiguous()
            source_value = frame.value[
                batch_index, token_start:token_end, head_index
            ].contiguous()
            use_pinned = bool(
                self.config.pin_memory
                and target_device.type == "cuda"
                and torch.cuda.is_available()
            )
            if use_pinned:
                pinned_key = torch.empty_like(source_key, pin_memory=True)
                pinned_value = torch.empty_like(source_value, pin_memory=True)
                pinned_key.copy_(source_key)
                pinned_value.copy_(source_value)
                source_key, source_value = pinned_key, pinned_value
            missing.append((cache_key, source_key, source_value, uses, batch_index))
        cpu_gather_s = time.perf_counter() - gather_start

        transfer_start = time.perf_counter()
        materialized_missing: list[
            tuple[CachedRawHistoryBlock, list[tuple[int, int]], int]
        ] = []
        for cache_key, source_key, source_value, uses, batch_index in missing:
            key_device = source_key.to(
                target_device,
                non_blocking=source_key.is_pinned()
                and self.config.non_blocking_h2d,
            )
            value_device = source_value.to(
                target_device,
                non_blocking=source_value.is_pinned()
                and self.config.non_blocking_h2d,
            )
            entry = CachedRawHistoryBlock(
                key=cache_key,
                key_unrotated=key_device,
                value=value_device,
            )
            cache.put(entry)
            materialized_missing.append((entry, uses, batch_index))
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        h2d_s = time.perf_counter() - transfer_start

        for entry, uses, batch_index in resident + materialized_missing:
            for union_index, local_token in uses:
                key_unrotated[
                    batch_index, union_index, entry.key.head_id
                ] = entry.key_unrotated[local_token]
                value[batch_index, union_index, entry.key.head_id] = entry.value[
                    local_token
                ]

        if candidate_frame_ids is None:
            position_candidates = torch.tensor(
                list(dict.fromkeys(int(value) for value in frames[frames >= 0])),
                dtype=torch.long,
            )
        elif isinstance(candidate_frame_ids, torch.Tensor):
            position_candidates = candidate_frame_ids.detach().to("cpu").long()
        else:
            position_candidates = torch.tensor(candidate_frame_ids, dtype=torch.long)
        positions = build_sparse_positions(
            frame_ids=route_plan.union_frame_ids.clamp_min(0),
            token_ids=route_plan.union_token_ids.clamp_min(0),
            current_frame_id=current_frame_id,
            spatial_width=self.spatial_width,
            rope_policy=self.config.rope_policy,
            max_relative_age=self.config.max_relative_age,
            candidate_frame_ids=position_candidates,
        )
        rope_start = time.perf_counter()
        key = key_unrotated
        if freqs is not None:
            key = apply_selected_rope(
                key_unrotated,
                positions.to(target_device),
                freqs.to(target_device),
            )
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
        rope_s = time.perf_counter() - rope_start
        miss_bytes = sum(entry.bytes for entry, _, _ in materialized_missing)
        hit_bytes = sum(entry.bytes for entry, _, _ in resident)
        logical_miss_tokens = sum(len(uses) for _, uses, _ in materialized_missing)
        logical_miss_bytes = logical_miss_tokens * 2 * dim * first.key.element_size()
        return MaterializedHistory(
            key_unrotated=key_unrotated,
            key=key,
            value=value,
            positions=positions,
            transferred_bytes=miss_bytes,
            cpu_gather_s=cpu_gather_s,
            h2d_s=h2d_s,
            rope_s=rope_s,
            payload_bytes=logical_miss_bytes,
            padding_bytes=max(0, miss_bytes - logical_miss_bytes),
            source_run_count=len(materialized_missing),
            cache_hit=not materialized_missing,
            h2d_copy_count=2 * len(materialized_missing),
            staging_mode="cross_chunk_raw_block64",
            cache_hit_bytes=hit_bytes,
            cache_miss_bytes=miss_bytes,
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

    @profiled("history/full_candidate_concatenation")
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
