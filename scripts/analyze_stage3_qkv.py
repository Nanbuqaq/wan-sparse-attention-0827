#!/usr/bin/env python3
"""Stage-3 captured-QKV retrieval and V-aware objective diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from adapters.routing import RoutingState, route_attention
from adapters.types import MethodConfig


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = candidate.float() - reference.float()
    relative = torch.linalg.vector_norm(delta, dim=-1) / torch.linalg.vector_norm(
        reference.float(), dim=-1
    ).clamp_min(1e-12)
    maximum = delta.abs().amax(dim=-1)
    return relative, maximum


def sample_query_ids(length: int, count: int, device: torch.device) -> torch.Tensor:
    frames, height, width = 21, 30, 52
    if frames * height * width != length:
        return torch.linspace(0, length - 1, count, device=device).round().long().unique()
    temporal = torch.linspace(0, frames - 1, 5, device=device).round().long()
    vertical = torch.tensor([4, 14, 25], device=device)
    horizontal = torch.tensor([6, 25, 45], device=device)
    grid = []
    for t in temporal.tolist():
        for y in vertical.tolist():
            for x in horizontal.tolist():
                grid.append(t * height * width + y * width + x)
    ids = torch.tensor(grid, device=device, dtype=torch.long)
    if ids.numel() > count:
        positions = torch.linspace(0, ids.numel() - 1, count, device=device).round().long()
        ids = ids.index_select(0, positions)
    return ids.unique()


def query_coordinates(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = 30, 52
    t = ids // (height * width)
    remainder = ids % (height * width)
    return t, remainder // width, remainder % width


def selected_original_mask(plan, head: int, query_ids: torch.Tensor) -> torch.Tensor:
    length = int(plan.metadata["original_length"])
    heads = plan.block_map.shape[1]
    row = head
    if plan.q_sorted_indices is None:
        q_work = query_ids
    else:
        order = plan.q_sorted_indices.reshape(-1, length)[row].long()
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(length, device=order.device)
        q_work = inverse.index_select(0, query_ids)
    q_edges = plan.q_sizes[0, head].cumsum(0).long()
    q_blocks = torch.bucketize(q_work, q_edges, right=True).clamp_max(
        plan.q_sizes.shape[-1] - 1
    )
    k_block_for_work_token = torch.repeat_interleave(
        torch.arange(plan.k_sizes.shape[-1], device=query_ids.device),
        plan.k_sizes[0, head].long(),
    )
    selected_work = plan.block_map[0, head].index_select(0, q_blocks)
    selected_work = selected_work.index_select(1, k_block_for_work_token)
    if plan.k_sorted_indices is None:
        return selected_work
    k_order = plan.k_sorted_indices.reshape(-1, length)[row].long()
    selected_original = torch.zeros_like(selected_work)
    selected_original[:, k_order] = selected_work
    return selected_original


def method_specs() -> list[dict]:
    formal = json.loads((ROOT / "configs/formal_stage2_v2.json").read_text())
    lookup = {item["id"]: item for item in formal["methods"]}
    ids = [
        "block",
        "fixed_k128",
        "capacity_balanced",
        "radius_adaptive",
        "hierarchical",
        "product_quantized",
        "spatiotemporal",
        "query_metric",
    ]
    rows = [{**lookup[item], "analysis_id": item} for item in ids]
    common = {
        "mode": "sparse",
        "backend": "fixed64_bf16",
        "parameter_origin": "stage3_captured_screen",
        "kmeans_init_iterations": 3,
        "kmeans_step_iterations": 1,
    }
    rows.extend(
        [
            {
                **common,
                "analysis_id": "coverage_b70_l15",
                "method": "coverage_cluster",
                "route_params": {"base_fraction": 0.70, "local_fraction": 0.15, "remote_clusters": 128, "refresh_calls": 10, "frames_latent": 21, "height_latent": 30, "width_latent": 52},
            },
            {
                **common,
                "analysis_id": "coverage_b80_l10",
                "method": "coverage_cluster",
                "route_params": {"base_fraction": 0.80, "local_fraction": 0.10, "remote_clusters": 128, "refresh_calls": 10, "frames_latent": 21, "height_latent": 30, "width_latent": 52},
            },
            {
                **common,
                "analysis_id": "vaware_prototype_b80",
                "method": "vaware_cluster",
                "route_params": {"base_fraction": 0.80, "local_fraction": 0.10, "remote_clusters": 128, "refresh_calls": 10, "v_objective": "v_prototype", "v_weight": 0.75, "frames_latent": 21, "height_latent": 30, "width_latent": 52},
            },
            {
                **common,
                "analysis_id": "vaware_residual_b80",
                "method": "vaware_cluster",
                "route_params": {"base_fraction": 0.80, "local_fraction": 0.10, "remote_clusters": 128, "refresh_calls": 10, "v_objective": "output_residual", "v_weight": 0.75, "frames_latent": 21, "height_latent": 30, "width_latent": 52},
            },
            {
                **common,
                "analysis_id": "hybrid_residual_b80",
                "method": "stage3_hybrid",
                "route_params": {"base_fraction": 0.80, "local_fraction": 0.10, "remote_clusters": 128, "refresh_calls": 20, "v_objective": "output_residual", "v_weight": 0.75, "early_base_bonus": 0.05, "late_base_bonus": 0.025, "frames_latent": 21, "height_latent": 30, "width_latent": 52},
            },
        ]
    )
    return rows


def config_from_spec(spec: dict, density: float) -> MethodConfig:
    return MethodConfig(
        method=spec["method"],
        backend=spec.get("backend", "fixed64_bf16"),
        density=density,
        parameter_origin=spec.get("parameter_origin", "stage3_captured_screen"),
        q_clusters=int(spec.get("q_clusters", 128)),
        k_clusters=int(spec.get("k_clusters", 128)),
        kmeans_init_iterations=int(spec.get("kmeans_init_iterations", 3)),
        kmeans_step_iterations=int(spec.get("kmeans_step_iterations", 1)),
        route_params=dict(spec.get("route_params", {})),
        inference_steps=50,
        calls_per_step=2,
    )


def exact_query_data(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ids: torch.Tensor):
    query = q.index_select(0, ids).float()
    scores = torch.matmul(query, k.float().transpose(0, 1)) / math.sqrt(q.shape[-1])
    probability = torch.softmax(scores, dim=-1)
    output = torch.matmul(probability, v.float())
    return probability, output


def route_rows_for_head(
    *,
    point: dict,
    spec: dict,
    plan,
    head_index: int,
    head_id: int,
    ids: torch.Tensor,
    probability: torch.Tensor,
    dense_output: torch.Tensor,
    value: torch.Tensor,
    graph_hash: str,
) -> list[dict]:
    mask = selected_original_mask(plan, head_index, ids)
    selected_probability = probability * mask.float()
    mass = selected_probability.sum(dim=-1)
    normalized = selected_probability / mass.unsqueeze(-1).clamp_min(1e-12)
    sparse_output = torch.matmul(normalized, value.float())
    relative, maximum = tensor_error(dense_output, sparse_output)
    value_norm = torch.linalg.vector_norm(value.float(), dim=-1)
    pv_total = (probability * value_norm.unsqueeze(0)).sum(dim=-1)
    pv_selected = (selected_probability * value_norm.unsqueeze(0)).sum(dim=-1)
    t, y, x = query_coordinates(ids)
    rows = []
    for index in range(ids.numel()):
        rows.append(
            {
                "analysis_id": spec["analysis_id"],
                "route_method": spec["method"],
                "layer": int(point["layer"]),
                "call_index": int(point["call_index"]),
                "head_id": int(head_id),
                "query_id": int(ids[index]),
                "query_time": int(t[index]),
                "query_y": int(y[index]),
                "query_x": int(x[index]),
                "attention_mass_recall": float(mass[index]),
                "pv_magnitude_recall": float(pv_selected[index] / pv_total[index].clamp_min(1e-12)),
                "output_relative_l2": float(relative[index]),
                "output_max_abs": float(maximum[index]),
                "selected_token_fraction": float(mask[index].float().mean()),
                "logical_density": float(plan.logical_density),
                "cluster_ms": float(plan.cluster_ms),
                "selection_ms": float(plan.selection_ms),
                "permutation_ms": float(plan.permutation_ms),
                "preserves_original_order": plan.q_sorted_indices is None and plan.k_sorted_indices is None,
                "route_graph_sha256": graph_hash,
            }
        )
    return rows


def objective_rows_for_head(
    *,
    point: dict,
    head_id: int,
    ids: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    probability: torch.Tensor,
    dense_output: torch.Tensor,
    density: float,
) -> list[dict]:
    length, dim = k.shape
    block_size = 64
    blocks = math.ceil(length / block_size)
    padded = blocks * block_size
    sizes = torch.full((blocks,), block_size, device=k.device, dtype=torch.float32)
    if length % block_size:
        sizes[-1] = length % block_size
    k_pad = F.pad(k.float(), (0, 0, 0, padded - length)).view(blocks, block_size, dim)
    v_pad = F.pad(v.float(), (0, 0, 0, padded - length)).view(blocks, block_size, dim)
    k_mean = k_pad.sum(dim=1) / sizes.unsqueeze(-1)
    v_mean = v_pad.sum(dim=1) / sizes.unsqueeze(-1)
    p_pad = F.pad(probability, (0, padded - length)).view(ids.numel(), blocks, block_size)
    p_mass = p_pad.sum(dim=-1)
    v_token_norm = torch.linalg.vector_norm(v_pad, dim=-1)
    p_vnorm = (p_pad * v_token_norm.unsqueeze(0)).sum(dim=-1)
    pv_vector = torch.einsum("nbt,btd->nbd", p_pad, v_pad).norm(dim=-1)
    v_norm = torch.linalg.vector_norm(v_mean, dim=-1).unsqueeze(0).expand_as(p_mass)
    qk_block = torch.matmul(q.index_select(0, ids).float(), k_mean.transpose(0, 1)) / math.sqrt(dim)
    residual = torch.linalg.vector_norm(
        v_mean.unsqueeze(0) - dense_output.unsqueeze(1), dim=-1
    )
    objectives = {
        "qk_block": qk_block,
        "p_mass": p_mass,
        "p_x_vnorm": p_vnorm,
        "pv_vector_norm": pv_vector,
        "v_norm_only": v_norm,
        "v_prototype": p_mass * v_norm,
        "output_residual_oracle": p_mass * residual,
    }
    budget = max(1, min(blocks, int(round(blocks * density))))
    rows = []
    t, y, x = query_coordinates(ids)
    for name, score in objectives.items():
        selected_blocks = torch.zeros_like(score, dtype=torch.bool)
        selected_blocks.scatter_(-1, torch.topk(score, k=budget, dim=-1).indices, True)
        token_mask = selected_blocks.unsqueeze(-1).expand(-1, -1, block_size).reshape(
            ids.numel(), padded
        )[:, :length]
        selected_probability = probability * token_mask.float()
        recall = selected_probability.sum(dim=-1)
        normalized = selected_probability / recall.unsqueeze(-1).clamp_min(1e-12)
        output = torch.matmul(normalized, v.float())
        relative, maximum = tensor_error(dense_output, output)
        for index in range(ids.numel()):
            rows.append(
                {
                    "objective": name,
                    "layer": int(point["layer"]),
                    "call_index": int(point["call_index"]),
                    "head_id": int(head_id),
                    "query_id": int(ids[index]),
                    "query_time": int(t[index]),
                    "query_y": int(y[index]),
                    "query_x": int(x[index]),
                    "attention_mass_recall": float(recall[index]),
                    "output_relative_l2": float(relative[index]),
                    "output_max_abs": float(maximum[index]),
                    "selected_token_fraction": float(token_mask[index].float().mean()),
                    "offline_oracle": name in {"p_mass", "p_x_vnorm", "pv_vector_norm", "output_residual_oracle"},
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for name, values in sorted(groups.items()):
        error = [float(item["output_relative_l2"]) for item in values]
        recall = [float(item["attention_mass_recall"]) for item in values]
        worst = max(values, key=lambda item: float(item["output_relative_l2"]))
        output.append(
            {
                key: name,
                "queries": len(values),
                "attention_mass_recall_mean": sum(recall) / len(recall),
                "output_relative_l2_mean": sum(error) / len(error),
                "output_relative_l2_p90": sorted(error)[int(0.9 * (len(error) - 1))],
                "output_relative_l2_max": max(error),
                "worst_layer": worst["layer"],
                "worst_call_index": worst["call_index"],
                "worst_head_id": worst["head_id"],
                "worst_query_id": worst["query_id"],
                "worst_query_time": worst["query_time"],
            }
        )
    return output


def make_plots(route_summary: list[dict], objective_summary: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in route_summary:
        ax.scatter(row["attention_mass_recall_mean"], row["output_relative_l2_mean"])
        ax.annotate(row["analysis_id"], (row["attention_mass_recall_mean"], row["output_relative_l2_mean"]), fontsize=7)
    ax.set_xlabel("sampled-query attention-mass recall")
    ax.set_ylabel("sampled-query output relative L2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "route_recall_vs_output_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [row["objective"] for row in objective_summary]
    values = [row["output_relative_l2_mean"] for row in objective_summary]
    ax.bar(names, values)
    ax.set_ylabel("sampled-query output relative L2")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "v_objective_output_error.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", default="results/captures/qkv_v2/chef_motion/seed_000042/capture_manifest.json")
    parser.add_argument("--max-points", type=int, default=12)
    parser.add_argument("--record-index", action="append", type=int, default=[])
    parser.add_argument("--sample-queries", type=int, default=36)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objectives-only", action="store_true")
    parser.add_argument("--output-dir", default="results/metrics/stage3_qkv_diagnostics")
    args = parser.parse_args()
    if not args.objectives_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for captured Stage-3 routing diagnostics")
    manifest_path = Path(args.capture_manifest)
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"][: args.max_points]
    if args.record_index:
        records = [manifest["records"][index] for index in args.record_index]
    specs = [] if args.objectives_only else method_specs()
    if args.include:
        specs = [item for item in specs if any(value in item["analysis_id"] for value in args.include)]
    states: dict[tuple[str, int], RoutingState] = {}
    route_rows = []
    objective_rows = []
    failures = []
    for record in records:
        point = torch.load(record["path"], map_location="cpu", weights_only=False)
        q = point["q"].to(args.device)
        k = point["k"].to(args.device)
        v = point["v"].to(args.device)
        ids = sample_query_ids(q.shape[2], args.sample_queries, q.device)
        exact = []
        for head_index in range(q.shape[1]):
            probability, dense_output = exact_query_data(q[0, head_index], k[0, head_index], v[0, head_index], ids)
            exact.append((probability, dense_output))
            objective_rows.extend(
                objective_rows_for_head(
                    point=point,
                    head_id=point["head_ids"][head_index],
                    ids=ids,
                    q=q[0, head_index],
                    k=k[0, head_index],
                    v=v[0, head_index],
                    probability=probability,
                    dense_output=dense_output,
                    density=args.density,
                )
            )
        for spec in specs:
            state = states.setdefault((spec["analysis_id"], int(point["layer"])), RoutingState())
            try:
                _, _, _, plan = route_attention(
                    q,
                    k,
                    v,
                    config=config_from_spec(spec, args.density),
                    state=state,
                    layer=int(point["layer"]),
                    call_index=int(point["call_index"]),
                )
                graph_hash = plan.graph_sha256()
                for head_index, (probability, dense_output) in enumerate(exact):
                    route_rows.extend(
                        route_rows_for_head(
                            point=point,
                            spec=spec,
                            plan=plan,
                            head_index=head_index,
                            head_id=point["head_ids"][head_index],
                            ids=ids,
                            probability=probability,
                            dense_output=dense_output,
                            value=v[0, head_index],
                            graph_hash=graph_hash,
                        )
                    )
            except Exception as error:
                failures.append(
                    {
                        "analysis_id": spec["analysis_id"],
                        "point": record["path"],
                        "layer": point["layer"],
                        "call_index": point["call_index"],
                        "error": repr(error),
                    }
                )
        del q, k, v, exact
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "route_query_rows.csv", route_rows)
    write_csv(output_dir / "objective_query_rows.csv", objective_rows)
    route_summary = summarize(route_rows, "analysis_id")
    objective_summary = summarize(objective_rows, "objective")
    write_csv(output_dir / "route_summary.csv", route_summary)
    write_csv(output_dir / "objective_summary.csv", objective_summary)
    make_plots(route_summary, objective_summary, ROOT / "results/figures/stage3_qkv_diagnostics")
    payload = {
        "schema_version": 3,
        "capture_manifest": str(manifest_path.resolve()),
        "capture_points": len(records),
        "sample_queries_per_head": args.sample_queries,
        "density": args.density,
        "route_summary": route_summary,
        "objective_summary": objective_summary,
        "failures": failures,
        "objectives_only": args.objectives_only,
        "status": "pass" if objective_rows and (args.objectives_only or route_rows) and not failures else "fail",
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "capture_points": len(records), "route_rows": len(route_rows), "objective_rows": len(objective_rows), "failures": len(failures), "output": str(output_dir)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
