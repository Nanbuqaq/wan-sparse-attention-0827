import pytest
import torch

from scripts.probe_oracle_history_query_roles import group_means, role_metrics


def test_group_labels_do_not_reset_at_latent_frame_boundaries():
    tokens = torch.cat([torch.zeros(1560), torch.ones(1560)])
    groups = group_means(tokens)
    assert groups[24] == 40/64
    assert len(groups) == 49


def test_weighted_role_metric_counts_last_short_group_correctly():
    metric = role_metrics(torch.tensor([0., 1.]), torch.tensor([0., 0.]), torch.tensor([64, 8]))
    assert metric['soft_mask_mae'] == pytest.approx(8/72)
    assert metric['balanced_accuracy'] is None
    with pytest.raises(ValueError, match='finite'):
        role_metrics(torch.tensor([float('nan')]), torch.ones(1), torch.ones(1))
