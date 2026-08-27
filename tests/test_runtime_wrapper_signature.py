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

