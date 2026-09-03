from __future__ import annotations

import pandas as pd

from scripts.summarize_pareto_expansion import nondominated


def test_nondominated_keeps_quality_speed_tradeoff():
    rows = pd.DataFrame(
        [
            {
                "method": "quality",
                "ssim_mean": 0.7,
                "late_ssim_mean": 0.6,
                "lpips_mean": 0.3,
                "end_to_end_s_mean": 10.0,
                "transfer_density_mean": 0.25,
            },
            {
                "method": "speed",
                "ssim_mean": 0.65,
                "late_ssim_mean": 0.55,
                "lpips_mean": 0.31,
                "end_to_end_s_mean": 5.0,
                "transfer_density_mean": 0.25,
            },
            {
                "method": "dominated",
                "ssim_mean": 0.60,
                "late_ssim_mean": 0.50,
                "lpips_mean": 0.4,
                "end_to_end_s_mean": 12.0,
                "transfer_density_mean": 0.5,
            },
        ]
    )
    assert nondominated(rows) == ["quality", "speed"]
