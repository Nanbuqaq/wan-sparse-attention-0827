"""Task-scoped execution dependency manifests.

The reuse key deliberately hashes only code and runtime inputs that can affect
the concrete task.  Unrelated routes, reports, and plotting changes therefore
do not invalidate an existing video.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import platform
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
VENDORED_SVOO = ROOT / "adapters" / "vendor" / "svoo_repo" / "svoo"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))


def _normalized_symbol_source(module_name: str, symbol_name: str) -> tuple[str, str]:
    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name)
    source = inspect.getsource(symbol)
    tree = ast.parse(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    source_path = Path(inspect.getsourcefile(symbol) or inspect.getfile(symbol)).resolve()
    return str(source_path), sha256_bytes(normalized.encode("utf-8"))


def _file_entry(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "kind": "file",
        "label": label,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def _symbol_entry(module_name: str, symbol_name: str) -> dict[str, Any]:
    path, digest = _normalized_symbol_source(module_name, symbol_name)
    return {
        "kind": "symbol_ast",
        "module": module_name,
        "symbol": symbol_name,
        "path": path,
        "sha256": digest,
    }


ROUTE_SYMBOLS: dict[str, tuple[tuple[str, str], ...]] = {
    "original_block": (("adapters.routes.baselines", "route_original_block"), ("adapters.routing", "_route_attention_legacy"), ("adapters.routing", "_fixed_plan")),
    "random_block": (("adapters.routes.baselines", "route_random_block"), ("adapters.routing", "_route_random_block"), ("adapters.routing", "_fixed_plan")),
    "local_3d": (("adapters.routes.baselines", "route_local_3d"), ("adapters.routing", "_route_local_3d"), ("adapters.routing", "_fixed_plan")),
    "fixed_k128": (("adapters.routes.baselines", "route_fixed_k"), ("adapters.routing", "_route_attention_legacy"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "fixed_k256": (("adapters.routes.baselines", "route_fixed_k"), ("adapters.routing", "_route_attention_legacy"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "qsort_local8": (("adapters.routes.baselines", "route_qsort_local8"), ("adapters.routing", "_route_qsort_local8"), ("adapters.routing", "_fixed_plan")),
    "token_oracle": (("adapters.routes.baselines", "route_token_oracle"), ("adapters.routing", "_oracle_fixed_plan")),
    "svg2": (("adapters.routes.papers", "route_svg2"), ("adapters.routing", "_route_attention_legacy"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "calibrated_top_p_map"), ("adapters.routing", "_cluster_route_plan")),
    "svg2_fixed": (("adapters.routes.papers", "route_svg2"), ("adapters.routing", "_route_attention_legacy"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "calibrated_top_p_map"), ("adapters.routing", "_fixed_plan")),
    "svg2_varlen": (("adapters.routes.papers", "route_svg2"), ("adapters.routing", "_route_attention_legacy"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "calibrated_top_p_map"), ("adapters.routing", "_plan_metrics")),
    "svg2_official_top_p": (("adapters.routes.papers", "route_svg2"), ("adapters.routing", "_route_attention_legacy"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "top_p_map")),
    "adacluster": (("adapters.routes.papers", "route_adacluster"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "exact_pair_budget_map"), ("adapters.routing", "_cluster_route_plan"), ("adapters.routing", "_fixed_plan")),
    "svoo": (("adapters.routes.papers", "route_svoo"), ("adapters.routing", "exact_pair_budget_map"), ("adapters.routing", "_cluster_route_plan"), ("adapters.routing", "_fixed_plan")),
    "scope": (("adapters.routes.papers", "route_scope"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "capacity_balanced": (("adapters.routes.self_cluster", "route_capacity_balanced"), ("adapters.routes.self_cluster", "_timed_kmeans"), ("adapters.routes.self_cluster", "_finish"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "radius_adaptive": (("adapters.routes.self_cluster", "route_radius_adaptive"), ("adapters.routes.self_cluster", "_timed_kmeans"), ("adapters.routes.self_cluster", "_finish"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "hierarchical": (("adapters.routes.self_cluster", "route_hierarchical"), ("adapters.routes.self_cluster", "_timed_kmeans"), ("adapters.routes.self_cluster", "_finish"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "product_quantized": (("adapters.routes.self_cluster", "route_product_quantized"), ("adapters.routes.self_cluster", "_timed_kmeans"), ("adapters.routes.self_cluster", "_finish"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "spatiotemporal": (("adapters.routes.self_cluster", "route_spatiotemporal"), ("adapters.routes.self_cluster", "_timed_kmeans"), ("adapters.routes.self_cluster", "_finish"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
    "query_metric": (("adapters.routes.self_cluster", "route_query_metric"), ("adapters.routes.self_cluster", "_timed_kmeans"), ("adapters.routes.self_cluster", "_finish"), ("adapters.routing", "batched_euclidean_kmeans"), ("adapters.routing", "_fixed_plan")),
}

CLUSTER_METHODS = {
    "fixed_k128",
    "fixed_k256",
    "svg2",
    "svg2_fixed",
    "svg2_varlen",
    "svg2_official_top_p",
    "adacluster",
    "svoo",
    "scope",
    "capacity_balanced",
    "radius_adaptive",
    "hierarchical",
    "product_quantized",
    "spatiotemporal",
    "query_metric",
}


BACKEND_SYMBOLS: dict[str, tuple[str, str]] = {
    "fixed64_bf16": ("adapters.kernels_fixed64", "fixed64_sparse_attention"),
    "varlen_triton": ("adapters.kernels", "execute_route"),
    "varlen_triton_native": ("adapters.kernels", "execute_route"),
    "varlen_triton_csr": ("adapters.kernels_varlen_csr", "varlen_csr_attention"),
}


def generation_fingerprint(task: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model",
        "height",
        "width",
        "frames",
        "steps",
        "guidance",
        "shift",
        "fps",
    )
    values = {key: task.get(key, common.get(key)) for key in keys}
    values.update({"prompt": task["prompt"], "seed": int(task["seed"])})
    return values


def task_fingerprint(task: dict[str, Any], common: dict[str, Any]) -> str:
    payload = {
        "task": {key: value for key, value in task.items() if key != "output"},
        "generation": generation_fingerprint(task, common),
        "output": task.get("output"),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def build_execution_dependency_manifest(
    task: dict[str, Any],
    common: dict[str, Any],
    *,
    pipeline_class: type,
    scheduler_class: type,
) -> dict[str, Any]:
    model_root = Path(task.get("model", common["model"]))
    model_files = [
        model_root / "model_index.json",
        model_root / "transformer" / "config.json",
        model_root / "scheduler" / "scheduler_config.json",
    ]
    dependencies: list[dict[str, Any]] = [
        _symbol_entry("scripts.run_matrix", "run_generation_task"),
        _file_entry(Path(inspect.getsourcefile(pipeline_class) or inspect.getfile(pipeline_class)), label="pipeline"),
        _file_entry(Path(inspect.getsourcefile(scheduler_class) or inspect.getfile(scheduler_class)), label="scheduler"),
    ]
    dependencies.extend(
        _file_entry(path, label=f"model:{path.name}") for path in model_files if path.is_file()
    )
    if task["mode"] == "sparse":
        method = task["method"]
        backend = task["backend"]
        try:
            route_symbols = ROUTE_SYMBOLS[method]
        except KeyError as error:
            raise ValueError(f"no dependency mapping for route {method!r}") from error
        try:
            backend_module, backend_symbol = BACKEND_SYMBOLS[backend]
        except KeyError as error:
            raise ValueError(f"no dependency mapping for backend {backend!r}") from error
        dependencies.extend(
            [
                _symbol_entry("adapters.wan_sparse", "WanUnifiedSparseAttnProcessor"),
                _symbol_entry(backend_module, backend_symbol),
            ]
        )
        dependencies.extend(
            _symbol_entry(route_module, route_symbol)
            for route_module, route_symbol in route_symbols
        )
        if method in CLUSTER_METHODS:
            dependencies.extend(
                [
                    _file_entry(VENDORED_SVOO / "co_clustering.py", label="pinned_cluster_core"),
                    _file_entry(VENDORED_SVOO / "kernels" / "triton" / "permute.py", label="pinned_permutation"),
                    _file_entry(VENDORED_SVOO / "kernels" / "triton" / "l2norm.py", label="pinned_l2norm"),
                ]
            )
    dependencies = sorted(dependencies, key=lambda item: canonical_json(item))
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "diffusers": package_version("diffusers"),
        "triton": package_version("triton"),
        "pipeline_class": f"{pipeline_class.__module__}.{pipeline_class.__qualname__}",
        "scheduler_class": f"{scheduler_class.__module__}.{scheduler_class.__qualname__}",
    }
    payload = {
        "schema_version": 2,
        "task_fingerprint": task_fingerprint(task, common),
        "generation": generation_fingerprint(task, common),
        "runtime": runtime,
        "dependencies": dependencies,
    }
    payload["task_execution_hash"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return payload
