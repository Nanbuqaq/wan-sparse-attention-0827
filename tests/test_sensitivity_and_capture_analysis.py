from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import torch

from adapters.longlive_sparse import HistoryArchive, SparseHistoryConfig
from adapters.longlive_sparse.sensitivity import history_head_sensitivity
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer


def test_history_head_sensitivity_is_chunked_and_per_head() -> None:
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(1, 7, 2, 4, generator=generator)
    key = torch.randn(1, 9, 2, 4, generator=generator)
    value = torch.randn(1, 9, 2, 4, generator=generator)
    records = history_head_sensitivity(query, key, value, query_chunk_size=3)
    assert len(records) == 2
    assert records[0]["query_tokens"] == 7
    assert records[0]["history_tokens"] == 9
    assert records[0]["history_output_rms"] > 0


def test_archive_exposes_only_compact_online_context() -> None:
    config = SparseHistoryConfig(
        method="transfer_vaware_hybrid_history",
        history_density=0.25,
        block_size=4,
        method_params={
            "base_fraction": 0.7,
            "local_fraction": 0.15,
            "query_block_size": 4,
            "v_weight": 1.0,
            "transfer_multiplier": 1.0,
        },
    )
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    for frame_id in (2, 4):
        key = torch.randn(1, 8, 2, 4)
        archive.index_frame(0, frame_id, key, key.clone())
    summary = summarize_query_for_pretransfer(torch.randn(1, 8, 2, 4), 4)
    context = archive.online_routing_context(0, summary, [2, 4])
    assert context.blocks == 4
    assert context.metadata["raw_candidate_kv_exposed"] is False


def test_capture_analysis_script_parses_and_has_help() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/analyze_capture_system.py"
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    subprocess.run(["/usr/bin/python3", str(script), "--help"], check=True)
