from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


@pytest.mark.parametrize('layout', ['exact_compact', 'block64', 'page256', 'frame1560'])
@pytest.mark.parametrize('fused', [False, True])
def test_archive_runs_match_gather_without_concatenating_candidates(layout, fused, monkeypatch):
    archive = HistoryArchive(SparseHistoryConfig(method='rag_dense'), spatial_height=1, spatial_width=130)
    for frame in [9, 3]:
        # Deliberately strided original archive source and B>1/H>1.
        key = torch.randn(2, 260, 2, 4)[:, ::2]
        archive._layers.setdefault(0, {})[frame] = SimpleNamespace(key=key, value=key + 1)
    frames = torch.tensor([9, 3]).repeat_interleave(130).view(1, 1, -1).expand(2, 2, -1)
    tokens = torch.arange(130).repeat(2).view(1, 1, -1).expand(2, 2, -1)
    route = build_route_plan(method='fixture', routing_stage='pre-transfer',
        query_labels=torch.zeros(2, 2, 2, dtype=torch.long),
        selections=[[[torch.tensor([0, 128, 129, 130, 259])], [torch.tensor([129, 130])]],
                    [[torch.tensor([1, 2, 131])], [torch.tensor([128])]]],
        history_frame_ids=frames, history_token_ids=tokens,
        candidate_history_tokens=260, exact_k_tokens=0, density=.25, metadata={})
    plan = build_transfer_plan(route, [9, 3], frame_tokens=130, layout=layout, bytes_per_token=32)
    expected = archive.materialize_transfer_plan(0, plan, route, device='cpu',
                                                current_frame_id=10, freqs=None)
    def no_full_gather(*args, **kwargs):
        raise AssertionError('optimized pack concatenated the full candidate history')
    monkeypatch.setattr(archive, 'dense_history_tensors', no_full_gather)
    pool = PinnedStagingPool(slots=1, budget_bytes=1024**2, pin_memory=False)
    for repeat in range(2):
        actual = archive.materialize_transfer_plan(0, plan, route, device='cpu',
            current_frame_id=10, freqs=None, staging_pool=pool,
            staging_mode='persistent_fused' if fused else 'persistent_separate', cpu_pack_policy='archive_runs')
        torch.testing.assert_close(actual.key, expected.key, rtol=0, atol=0)
        torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
        assert actual.transferred_bytes == expected.transferred_bytes
        assert actual.staging_reused == bool(repeat)
        assert actual.materialize_total_s >= actual.cpu_gather_s + actual.h2d_s
