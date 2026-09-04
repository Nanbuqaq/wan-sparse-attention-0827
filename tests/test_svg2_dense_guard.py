from __future__ import annotations

import pytest

from adapters.types import MethodConfig
from adapters.wan_sparse import resolve_svg2_dense_guard, svg2_dense_guard_policy


def make_config(method: str, guard: bool | None, *, steps: int = 50) -> MethodConfig:
    return MethodConfig(
        method=method,
        backend="varlen_triton",
        svg2_dense_guard=guard,
        inference_steps=steps,
        calls_per_step=2,
    )


def test_none_preserves_historical_method_defaults() -> None:
    assert resolve_svg2_dense_guard(make_config("svg2_official_top_p", None)) is True
    assert resolve_svg2_dense_guard(make_config("svg2_varlen", None)) is False


def test_explicit_guard_overrides_both_svg2_selection_policies() -> None:
    assert resolve_svg2_dense_guard(make_config("svg2_varlen", True)) is True
    assert resolve_svg2_dense_guard(make_config("svg2_official_top_p", False)) is False


def test_wan_1p3b_floor_policy_has_no_dense_layer() -> None:
    policy = svg2_dense_guard_policy(make_config("svg2_varlen", True), num_layers=30)
    assert policy["dense_steps"] == 10
    assert policy["dense_layers"] == 0
    assert policy["expected_explicit_dense_calls"] == 600
    assert policy["expected_sparse_calls"] == 2400
    assert policy["expected_total_calls"] == 3000


def test_five_step_smoke_policy_counts() -> None:
    guarded = svg2_dense_guard_policy(
        make_config("svg2_varlen", True, steps=5), num_layers=30
    )
    unguarded = svg2_dense_guard_policy(
        make_config("svg2_official_top_p", False, steps=5), num_layers=30
    )
    assert guarded["dense_steps"] == 1
    assert guarded["dense_layers"] == 0
    assert guarded["expected_explicit_dense_calls"] == 60
    assert guarded["expected_sparse_calls"] == 240
    assert unguarded["expected_explicit_dense_calls"] == 0
    assert unguarded["expected_sparse_calls"] == 300


def test_upstream_40_layer_fraction_maps_to_one_layer() -> None:
    policy = svg2_dense_guard_policy(
        make_config("svg2_official_top_p", None), num_layers=40
    )
    assert policy["dense_layers"] == 1


def test_guard_rejects_non_svg2_method() -> None:
    with pytest.raises(ValueError, match="only valid for SVG2"):
        MethodConfig(method="original_block", svg2_dense_guard=True)
