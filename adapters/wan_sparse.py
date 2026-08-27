"""Diffusers Wan self-attention processor for the unified sparse methods."""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

from .kernels import execute_route
from .routing import RoutingState, route_attention
from .types import MethodConfig, SparseRunStats
from .vendor import source_hashes


_LAYER_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.attn1\.processor$")


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _project_qkv(attn, hidden_states: torch.Tensor):
    if getattr(attn, "fused_projections", False):
        return attn.to_qkv(hidden_states).chunk(3, dim=-1)
    return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)


class WanUnifiedSparseAttnProcessor:
    def __init__(
        self,
        *,
        layer: int,
        num_layers: int,
        config: MethodConfig,
        stats: SparseRunStats,
    ) -> None:
        self.layer = layer
        self.num_layers = num_layers
        self.config = config
        self.stats = stats
        self.state = RoutingState()
        self.call_index = 0

    def _official_dense_reference(self) -> bool:
        if self.config.method != "svg2_official_top_p":
            return False
        dense_layers = math.floor(self.num_layers * self.config.official_first_layer_fraction)
        if self.layer < dense_layers:
            return True
        step = self.call_index // self.config.calls_per_step
        dense_steps = math.floor(
            self.config.inference_steps * self.config.official_first_timestep_fraction
        )
        return step < dense_steps

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb=None,
        **_kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise RuntimeError("unified sparse processor must only replace Wan self-attention")
        if attention_mask is not None:
            raise ValueError("unified sparse processor does not support attention masks")
        if hidden_states.dtype is not torch.bfloat16:
            raise TypeError(f"Wan sparse path requires BF16 hidden states, got {hidden_states.dtype}")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            query, key, value = _project_qkv(attn, hidden_states)
            query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
            key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)
                key = _apply_rotary_emb(key, *rotary_emb)
            query_bhld = query.transpose(1, 2).contiguous()
            key_bhld = key.transpose(1, 2).contiguous()
            value_bhld = value.transpose(1, 2).contiguous()

            if self._official_dense_reference():
                sparse_output = F.scaled_dot_product_attention(
                    query_bhld,
                    key_bhld,
                    value_bhld,
                    dropout_p=0.0,
                    is_causal=False,
                )
                total_pairs = (
                    query_bhld.shape[0]
                    * query_bhld.shape[1]
                    * query_bhld.shape[2]
                    * key_bhld.shape[2]
                )
                self.stats.record_dense_reference(
                    self.config.method,
                    total_pairs=total_pairs,
                )
            else:
                q_work, k_work, v_work, plan = route_attention(
                    query_bhld,
                    key_bhld,
                    value_bhld,
                    config=self.config,
                    state=self.state,
                    layer=self.layer,
                    call_index=self.call_index,
                )
                if self.config.route_params.get("record_route_graph_hash") and self.call_index == 0:
                    self.stats.route_graph_hashes[f"layer_{self.layer:02d}"] = plan.graph_sha256()
                sparse_output, kernel_ms, inverse_ms = execute_route(
                    q_work,
                    k_work,
                    v_work,
                    plan,
                )
                end_event.record()
                torch.cuda.synchronize()
                attention_ms = float(start_event.elapsed_time(end_event))
                self.stats.record_plan(
                    plan,
                    kernel_ms=kernel_ms,
                    inverse_ms=inverse_ms,
                    attention_ms=attention_ms,
                )

            output = sparse_output.transpose(1, 2).contiguous().flatten(2, 3).type_as(query)
            output = attn.to_out[0](output)
            return attn.to_out[1](output)
        except Exception:
            self.stats.failed_calls += 1
            raise
        finally:
            self.call_index += 1


def install_sparse_processors(
    transformer,
    *,
    config: MethodConfig,
    stats: SparseRunStats | None = None,
) -> SparseRunStats:
    stats = stats or SparseRunStats()
    stats.source_hashes.update(source_hashes())
    processors = dict(transformer.attn_processors)
    num_layers = len(transformer.blocks)
    replaced = 0
    for name in list(processors):
        match = _LAYER_PATTERN.search(name)
        if match is None:
            continue
        layer = int(match.group(1))
        processors[name] = WanUnifiedSparseAttnProcessor(
            layer=layer,
            num_layers=num_layers,
            config=config,
            stats=stats,
        )
        replaced += 1
    if replaced != num_layers:
        raise RuntimeError(f"expected {num_layers} Wan self-attention processors, replaced {replaced}")
    transformer.set_attn_processor(processors)
    return stats
