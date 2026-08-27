from __future__ import annotations

import math

import pytest
import torch

from adapters.kernels import execute_route
from adapters.routing import RoutingState, inverse_permute, route_attention
from adapters.types import MethodConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _element_mask(plan) -> torch.Tensor:
    batch, heads, _, _ = plan.block_map.shape
    length = int(plan.q_sizes[0, 0].sum())
    output = torch.zeros((batch, heads, length, length), device=plan.block_map.device, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            q_edges = torch.cat(
                (torch.zeros(1, device=output.device, dtype=torch.long), plan.q_sizes[b, h].cumsum(0).long())
            )
            k_edges = torch.cat(
                (torch.zeros(1, device=output.device, dtype=torch.long), plan.k_sizes[b, h].cumsum(0).long())
            )
            for qi in range(plan.q_sizes.shape[-1]):
                for ki in range(plan.k_sizes.shape[-1]):
                    if bool(plan.block_map[b, h, qi, ki]):
                        output[b, h, q_edges[qi] : q_edges[qi + 1], k_edges[ki] : k_edges[ki + 1]] = True
    return output


def _reference(q, k, v, plan):
    length = int(plan.metadata["original_length"])
    q = q[:, :, :length]
    k = k[:, :, :length]
    v = v[:, :, :length]
    mask = _element_mask(plan)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    output = torch.matmul(torch.softmax(scores.masked_fill(~mask, -float("inf")), dim=-1), v.float()).to(q.dtype)
    if plan.q_sorted_indices is not None:
        output = inverse_permute(output, plan.q_sorted_indices)
    return output


@pytest.mark.parametrize(
    "method,backend,q_clusters,k_clusters",
    [
        ("original_block", "fixed64_bf16", 4, 8),
        ("svg2_fixed", "fixed64_bf16", 4, 8),
        ("svg2_varlen", "varlen_triton", 4, 8),
        ("svg2_varlen", "varlen_triton_csr", 4, 8),
    ],
)
def test_sparse_kernel_matches_masked_dense(method, backend, q_clusters, k_clusters) -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, 256, 128, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    config = MethodConfig(
        method=method,
        backend=backend,
        density=0.40,
        q_clusters=q_clusters,
        k_clusters=k_clusters,
        kmeans_init_iterations=2,
        kmeans_step_iterations=1,
    )
    q_work, k_work, v_work, plan = route_attention(
        q, k, v, config=config, state=RoutingState(), layer=0, call_index=0
    )
    output, _, _ = execute_route(q_work, k_work, v_work, plan)
    reference = _reference(q_work, k_work, v_work, plan)
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "method,backend",
    [
        ("svg2_fixed", "fixed64_bf16"),
        ("svg2_varlen", "varlen_triton"),
        ("svg2_varlen", "varlen_triton_csr"),
    ],
)
def test_full_density_matches_dense(method, backend) -> None:
    torch.manual_seed(5)
    q = torch.randn(1, 2, 256, 128, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    config = MethodConfig(
        method=method,
        backend=backend,
        density=1.0,
        q_clusters=4,
        k_clusters=8,
        kmeans_init_iterations=2,
        kmeans_step_iterations=1,
    )
    q_work, k_work, v_work, plan = route_attention(
        q, k, v, config=config, state=RoutingState(), layer=0, call_index=0
    )
    output, _, _ = execute_route(q_work, k_work, v_work, plan)
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(output, dense, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "method,q_clusters,k_clusters,route_params",
    [
        ("random_block", 4, 8, {}),
        ("local_3d", 4, 8, {"frames_latent": 4, "height_latent": 8, "width_latent": 8}),
        ("qsort_local8", 4, 8, {}),
        ("capacity_balanced", 4, 8, {"clusters": 8, "capacity_factor": 1.5}),
        ("radius_adaptive", 4, 8, {"base_clusters": 4, "max_added_clusters": 4, "radius_threshold": 4.0}),
        ("hierarchical", 4, 8, {"coarse_clusters": 2, "branches": 2}),
        ("product_quantized", 4, 8, {"subspaces": 4, "codebook_clusters": 2}),
        ("spatiotemporal", 4, 8, {"clusters": 8, "frames_latent": 4, "height_latent": 8, "width_latent": 8}),
        ("query_metric", 4, 8, {"clusters": 8, "rank": 8, "basis_refresh_calls": 2}),
        ("adacluster", 4, 8, {"q_clusters": 4, "initial_k_clusters": 4, "max_added_clusters": 4, "distance_threshold": 4.0}),
        ("scope", 4, 4, {"q_clusters": 4, "subspace_clusters": 4}),
        ("svoo", 4, 8, {"q_clusters": 4, "k_clusters": 8, "co_cluster_iterations": 1}),
    ],
)
def test_all_route_families_match_their_masked_dense(
    method, q_clusters, k_clusters, route_params
) -> None:
    torch.manual_seed(11)
    q = torch.randn(1, 1, 256, 128, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    config = MethodConfig(
        method=method,
        backend="fixed64_bf16",
        density=0.40,
        q_clusters=q_clusters,
        k_clusters=k_clusters,
        kmeans_init_iterations=2,
        kmeans_step_iterations=1,
        route_params=route_params,
    )
    q_work, k_work, v_work, plan = route_attention(
        q, k, v, config=config, state=RoutingState(), layer=0, call_index=0
    )
    output, _, _ = execute_route(q_work, k_work, v_work, plan)
    reference = _reference(q_work, k_work, v_work, plan)
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    assert plan.logical_pairs > 0
    assert plan.graph_sha256()


def test_adacluster_reuse_keeps_valid_route() -> None:
    torch.manual_seed(19)
    q = torch.randn(1, 1, 256, 128, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    state = RoutingState()
    config = MethodConfig(
        method="adacluster",
        backend="fixed64_bf16",
        density=0.25,
        q_clusters=4,
        k_clusters=8,
        kmeans_init_iterations=2,
        kmeans_step_iterations=1,
        route_params={
            "q_clusters": 4,
            "initial_k_clusters": 4,
            "max_added_clusters": 4,
            "distance_threshold": 4.0,
            "reuse_calls": 20,
        },
    )
    first = route_attention(q, k, v, config=config, state=state, layer=0, call_index=0)
    second = route_attention(q, k, v, config=config, state=state, layer=0, call_index=1)
    assert first[3].metadata["refreshed"] is True
    assert second[3].metadata["refreshed"] is False
    output, _, _ = execute_route(*second[:3], second[3])
    reference = _reference(*second[:3], second[3])
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


def test_same_route_graph_across_backends() -> None:
    torch.manual_seed(23)
    q = torch.randn(1, 1, 256, 128, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def build(backend, materialization=None):
        route_params = {}
        if materialization is not None:
            route_params["materialization"] = materialization
        config = MethodConfig(
            method="svg2",
            backend=backend,
            density=0.40,
            q_clusters=4,
            k_clusters=8,
            kmeans_init_iterations=2,
            kmeans_step_iterations=1,
            route_params=route_params,
            backend_params={"block_m": 64, "block_n": 32},
        )
        return route_attention(q, k, v, config=config, state=RoutingState(), layer=0, call_index=0)

    fixed = build("fixed64_bf16")
    fixed_native = build("varlen_triton_native", "fixed64_graph")
    fixed_csr = build("varlen_triton_csr", "fixed64_graph")
    assert fixed[3].graph_sha256() == fixed_native[3].graph_sha256() == fixed_csr[3].graph_sha256()

    varlen_native = build("varlen_triton_native")
    varlen_csr = build("varlen_triton_csr")
    assert varlen_native[3].graph_sha256() == varlen_csr[3].graph_sha256()
    for item in (fixed, fixed_native, fixed_csr, varlen_native, varlen_csr):
        output, _, _ = execute_route(*item[:3], item[3])
        reference = _reference(*item[:3], item[3])
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
