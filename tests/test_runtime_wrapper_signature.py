from __future__ import annotations

import ast
from pathlib import Path


def test_sparse_wrapper_accepts_rag_memory_indices():
    path = Path(__file__).resolve().parents[1] / "adapters/longlive_sparse/runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    assert methods
    assert any("memory_indices" in [argument.arg for argument in method.args.args] for method in methods)


def test_profile_captures_are_case_scoped_and_reset_per_inference() -> None:
    root = Path(__file__).resolve().parents[1]
    attention = (root / "adapters/longlive_sparse/runtime_attention.py").read_text(
        encoding="utf-8"
    )
    runtime = (root / "adapters/longlive_sparse/runtime.py").read_text(
        encoding="utf-8"
    )
    runner = (root / "scripts/run_loaded_method_suite.py").read_text(
        encoding="utf-8"
    )
    assert "LONGLIVE_CAPTURE_CASE_TAG" in attention
    assert "LONGLIVE_CAPTURE_ROUTE_LAYERS" in attention
    assert "module.clear_capture_state()" in runtime
    assert 'os.environ["LONGLIVE_CAPTURE_CASE_TAG"] = case_id' in runner
    assert 'os.environ.pop("LONGLIVE_CAPTURE_CASE_TAG", None)' in runner


def test_runtime_builds_new_utility_route_and_postprocesses_top_p_union() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "adapters/longlive_sparse/runtime_attention.py").read_text(
        encoding="utf-8"
    )
    assert 'self.sparse_config.method == "system_utility_history"' in source
    assert "route_system_utility" in source
    assert "apply_query_group_policy" in source
    assert "self.system_config.group_selection_policy" in source
