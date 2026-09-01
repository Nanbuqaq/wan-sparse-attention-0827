from __future__ import annotations

import hashlib

import numpy as np

from scripts.evaluate_videos import lpips_distances


def test_lpips_rejects_weight_sha_before_importing_optional_package(tmp_path):
    weights = tmp_path / "lpips.pth"
    trunk = tmp_path / "alexnet.pth"
    weights.write_bytes(b"fixed-private-weights")
    trunk.write_bytes(b"fixed-trunk-weights")
    video = np.zeros((1, 2, 2, 3), dtype=np.float32)
    values, error, provenance = lpips_distances(
        video,
        video,
        weights_path=weights,
        expected_sha256="0" * 64,
        expected_version="0.1.4",
        trunk_weights_path=trunk,
        expected_trunk_sha256=hashlib.sha256(trunk.read_bytes()).hexdigest(),
        expected_torch_version="2.7.0",
        expected_torchvision_version="0.22.0",
    )
    assert values is None
    assert "SHA mismatch" in error
    assert provenance["linear_weights_sha256"] != provenance[
        "expected_linear_weights_sha256"
    ]


def test_lpips_rejects_trunk_sha_before_importing_optional_package(tmp_path):
    weights = tmp_path / "lpips.pth"
    trunk = tmp_path / "alexnet.pth"
    weights.write_bytes(b"fixed-private-weights")
    trunk.write_bytes(b"fixed-trunk-weights")
    video = np.zeros((1, 2, 2, 3), dtype=np.float32)
    values, error, provenance = lpips_distances(
        video,
        video,
        weights_path=weights,
        expected_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        expected_version="0.1.4",
        trunk_weights_path=trunk,
        expected_trunk_sha256="0" * 64,
        expected_torch_version="2.7.0",
        expected_torchvision_version="0.22.0",
    )
    assert values is None
    assert "trunk weights SHA mismatch" in error
    assert provenance["trunk_weights_sha256"] != provenance[
        "expected_trunk_weights_sha256"
    ]
