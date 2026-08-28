from __future__ import annotations

from scripts.build_dense_screen_expected import SPECS


def test_dense_screen_runtime_specs_cover_both_dense_references():
    assert set(SPECS) == {"native_dense", "rag_dense"}
    assert SPECS["native_dense"]["routing_stage"] == "N/A"
    assert SPECS["rag_dense"]["routing_stage"] == "post-transfer"
    assert SPECS["native_dense"]["backend"] == "packed_fa2"
    assert SPECS["rag_dense"]["backend"] == "packed_fa2"
