import torch
from scripts.benchmark_dataflow_matrix import cases, selection


def test_matrix_is72_points_with_exact_union_and_all_query_groups():
    grid = cases()
    assert len(grid) == 72
    assert len({tuple(c.items()) for c in grid}) == 72
    for c in grid:
        indices, tags = selection(c)
        assert len(indices) == round(c['history_frames']*1560*c['density'])
        assert indices.unique().numel() == indices.numel()
        assert indices.min() >= 0 and indices.max() < c['history_frames']*1560
        assert set(tags.tolist()) == {0, 1, 2}
        assert torch.equal(tags, ((indices//1560)*25+(indices%1560)//64).int()%3)
