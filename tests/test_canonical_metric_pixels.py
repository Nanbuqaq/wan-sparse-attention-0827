from pathlib import Path
import numpy as np
import pytest
from scripts.evaluate_canonical_video_quality import metric_pixels
from scripts.evaluate_videos import psnr, ssim, lpips_distances


def test_canonical_uint8_is_normalized_before_shared_metrics():
    raw = np.zeros((1, 12, 12, 3), dtype=np.uint8)
    raw[:, :, 6:] = 255
    normalized = metric_pixels(raw)
    assert normalized.dtype == np.float32
    assert normalized.min() == 0 and normalized.max() == 1
    assert psnr(normalized[0], normalized[0]) == 100
    assert ssim(normalized[0], normalized[0]) == pytest.approx(1.)
    assert psnr(np.zeros_like(normalized[0]), np.ones_like(normalized[0])) == pytest.approx(0.)
    with pytest.raises(ValueError, match='uint8'):
        metric_pixels(normalized)


@pytest.mark.parametrize('pixels', [np.ones((1, 2, 2, 3), dtype=np.uint8),
    np.full((1, 2, 2, 3), 255., dtype=np.float32),
    np.full((1, 2, 2, 3), np.nan, dtype=np.float32)])
def test_lpips_rejects_invalid_input_without_loading_weights(pixels, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError('invalid pixels reached LPIPS loading')
    monkeypatch.setattr('scripts.evaluate_videos._load_audited_lpips', forbidden)
    result, error, _ = lpips_distances(pixels, pixels, weights_path=Path('unused'), expected_sha256='',
        expected_version='', trunk_weights_path=Path('unused'), expected_trunk_sha256='',
        expected_torch_version='', expected_torchvision_version='')
    assert result is None and error
