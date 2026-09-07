import json
from pathlib import Path

import pytest

from scripts.package_tether_sources import package
from scripts import run_official_tether_batch as batch
from scripts.prepare_official_tether_assets import link_exact


def test_source_archive_roundtrip_preserves_code_license_and_omits_assets(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("value = 3\n")
    (source / "LICENSE").write_text("upstream license\n")
    (source / "assets").mkdir()
    (source / "assets/unused.md").write_text("large nonruntime asset")
    target = tmp_path / "repo/vendor/tethermem.tar.gz"
    package(source, target, "a" * 40, "https://example.invalid/locked-source")
    monkeypatch.setattr(batch, "ROOT", tmp_path / "repo")
    destination = tmp_path / "extracted"
    lock = batch.unpack("tethermem", destination)
    assert (destination / "run.py").read_bytes() == (source / "run.py").read_bytes()
    assert (destination / "LICENSE").is_file()
    assert not (destination / "assets").exists()
    assert lock["unchanged_source_files"]


def test_asset_setup_never_retargets_existing_link(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("first")
    second.write_text("second")
    link = tmp_path / "models/link"
    link_exact(link, first)
    link_exact(link, first)
    with pytest.raises(FileExistsError):
        link_exact(link, second)
    assert link.resolve() == first


def test_official_protocol_does_not_substitute_previous_backbone():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/system/tether_official_reproduction_v1.json").read_text())
    assert protocol["generator"] == "causal_forcing.pt"
    assert protocol["retrieval"] == "ae_latent_mem.pt"
    assert protocol["lora"] is False
    assert protocol["paper_results_reproduced"] is False
