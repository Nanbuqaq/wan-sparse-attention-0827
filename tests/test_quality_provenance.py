from __future__ import annotations

import numpy as np

from scripts.evaluate_videos import lpips_distances


def test_lpips_rejects_weight_sha_before_importing_optional_package(tmp_path):
    weights = tmp_path / "lpips.pth"
    weights.write_bytes(b"fixed-private-weights")
    video = np.zeros((1, 2, 2, 3), dtype=np.float32)
    values, error, provenance = lpips_distances(
        video,
        video,
        weights_path=weights,
        expected_sha256="0" * 64,
        expected_version="0.1.4",
    )
    assert values is None
    assert "SHA mismatch" in error
    assert provenance["weights_sha256"] != provenance["expected_weights_sha256"]
