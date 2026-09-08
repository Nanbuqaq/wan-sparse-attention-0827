"""Configuration contract for sparse LongLive history attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .methods import BACKENDS, METHOD_SPECS, method_spec

_REFRESH_POLICIES = {"per_step", "per_chunk"}
_ROPE_POLICIES = {"upstream_zero", "recency_rank", "clipped_relative_age"}


@dataclass(frozen=True)
class SparseHistoryConfig:
    """All decisions that affect sparse history selection or execution.

    ``history_density`` is applied only to the coarse-retrieved historical
    candidate set.  Sink, current chunk, and the configured recent window are
    always exact and are accounted for separately in global executed density.
    """

    method: str = "block64_history"
    backend: str = "grouped_fa2"
    history_density: float = 0.25
    recent_exact_frames: int = 3
    block_size: int = 64
    clusters_per_frame: int = 32
    kmeans_iterations: int = 5
    kmeans_tolerance: float = 1e-4
    seed: int = 42
    refresh_policy: str = "per_step"
    rope_policy: str = "upstream_zero"
    max_relative_age: int = 31
    pin_memory: bool = True
    non_blocking_h2d: bool = True
    fail_on_fallback: bool = True
    record_per_call: bool = True
    method_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHOD_SPECS:
            raise ValueError(f"unsupported sparse history method: {self.method!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"unsupported sparse history backend: {self.backend!r}")
        if not 0.0 < self.history_density <= 1.0:
            raise ValueError("history_density must be in (0, 1]")
        if self.recent_exact_frames < 0:
            raise ValueError("recent_exact_frames must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.clusters_per_frame < 1:
            raise ValueError("clusters_per_frame must be positive")
        if self.kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")
        if self.kmeans_tolerance < 0:
            raise ValueError("kmeans_tolerance must be non-negative")
        if self.refresh_policy not in _REFRESH_POLICIES:
            raise ValueError(f"unsupported refresh_policy: {self.refresh_policy!r}")
        if self.rope_policy not in _ROPE_POLICIES:
            raise ValueError(f"unsupported rope_policy: {self.rope_policy!r}")
        if self.method=='whole_block_precision_history':
            if self.rope_policy!='upstream_zero' or self.refresh_policy!='per_chunk' or self.backend!='resident_grouped_fa2':
                raise ValueError('whole-block precision currently requires upstream-zero, per-chunk and resident baseline harness')
            count=self.method_params.get('precision_query_samples',1024)
            if type(count) is not int or count<1:raise ValueError('invalid precision query sample count')
            if self.method_params.get('precision_variance_codec','u8_scaled') not in ('bf16','u8_scaled'):
                raise ValueError('invalid precision variance codec')
            if self.method_params.get('precision_admission','mass_key_variance') not in ('mass_key_variance','mass_value','random'):
                raise ValueError('invalid precision admission')
        if self.method == 'group_relation_history':
            if self.rope_policy != 'upstream_zero':
                raise ValueError('group relation development currently requires upstream_zero')
            if self.method_params.get('information_grouping', 'spatial_quadrants') not in {
                    'spatial_quadrants', 'query_features', 'random_balanced'}:
                raise ValueError('invalid information_grouping')
            if self.method_params.get('relation_admission', 'per_group') not in {'shared', 'per_group'}:
                raise ValueError('invalid relation_admission')
            start = self.method_params.get('group_start_layer', 8)
            if not isinstance(start, int) or not 0 <= start <= 29:
                raise ValueError('group_start_layer must be layer0..29')
        if self.method in {'rope_aligned_final_history','rope_bootstrap_ablation_history'} and self.rope_policy != 'upstream_zero':
            raise ValueError('rope_aligned_final_history requires the validated upstream_zero policy')
        if self.method == 'rope_bootstrap_ablation_history':
            layer=self.method_params.get('bootstrap_layer',-1)
            if not isinstance(layer,int) or not -1 <= layer <= 29:
                raise ValueError('bootstrap_layer must be -1 (all) or a Wan1.3B layer0..29')
        if self.max_relative_age < 0:
            raise ValueError("max_relative_age must be non-negative")
        if self.method == 'tethermem_oracle_mask_teacher':
            if self.history_density != 1. or self.backend != 'split_role_sdpa_reference':
                raise ValueError('oracle teacher requires full KV transfer and split_role_sdpa_reference')
            if self.method_params.get('oracle_timeline') not in ('source_compatible_addressing', 'aligned_latent_anchors'):
                raise ValueError('oracle teacher requires explicit temporal addressing')
            for name in ('oracle_mask_sha256', 'oracle_reference_video_sha256'):
                value = self.method_params.get(name, '')
                if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('oracle teacher requires frozen mask and Dense reference SHA')
            if self.method_params.get('target_average', .25) != .25 or self.method_params.get('age_decay_floor', .05) != .05:
                raise ValueError('oracle teacher freezes public target and age defaults')
        allowed = set(method_spec(self.method).__dataclass_fields__)
        unknown = set(self.method_params) - allowed
        if unknown:
            raise ValueError(f"unknown method_params fields: {sorted(unknown)}")
        immutable = {"name", "category", "routing_stage", "counts_as_self_cluster"}
        forbidden = set(self.method_params) & immutable
        if forbidden:
            raise ValueError(
                f"method_params cannot change method identity: {sorted(forbidden)}"
            )

    @property
    def is_dense(self) -> bool:
        return self.method in {"native_dense", "rag_dense", "dense_history"} or self.history_density == 1.0

    def needs_bootstrap_raw(self, layer_id: int) -> bool:
        return self.method == 'rope_bootstrap_ablation_history' and self.method_params.get('bootstrap_layer',-1) in (-1,layer_id)

    def uses_aligned_prototypes(self, layer_id: int, candidate_frames: int) -> bool:
        if self.method=='rope_aligned_final_history':return True
        if self.method=='rope_bootstrap_ablation_history':
            return not (candidate_frames==1 and self.needs_bootstrap_raw(layer_id))
        return False

    @property
    def routing_stage(self) -> str:
        return method_spec(self.method).routing_stage

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SparseHistoryConfig":
        # OmegaConf/Mapping wrappers must not escape into JSON or weights-only
        # capture payloads. Parameter values are scalar MethodSpec fields.
        plain = dict(value or {})
        if "method_params" in plain:
            plain["method_params"] = dict(plain["method_params"] or {})
        return cls(**plain)
