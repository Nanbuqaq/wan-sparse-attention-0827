from __future__ import annotations

import numpy as np
import pytest

from scripts.build_video_review_storyboards import (
    expected_decoded_frames,
    quarter_sample_indices,
    runs,
)


def test_expected_frames_supports_basic_and_long_videos():
    assert expected_decoded_frames({"latent_frames": 120}) == 477
    assert expected_decoded_frames({"latent_frames": 240}) == 957
    with pytest.raises(ValueError):
        expected_decoded_frames({"latent_frames": 0})


def test_quarter_samples_cover_full_video_without_crossing_boundaries():
    quarters = quarter_sample_indices(477, 32)
    assert len(quarters) == 4
    assert all(len(indices) == 32 for indices in quarters)
    assert quarters[0][0] == 0
    assert quarters[-1][-1] == 476
    assert all(first[-1] < second[0] for first, second in zip(quarters, quarters[1:]))


def test_runs_keeps_only_contiguous_regions_at_or_above_minimum():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    assert runs(mask, minimum=3) == [(4, 7), (9, 11)]
