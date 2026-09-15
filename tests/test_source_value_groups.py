import torch
from adapters.longlive_sparse.source_value_groups import summarize_source_values,fill_grouped_source_values


def test_temporal_and_spatial_groups_have_equal_capacity_but_different_information():
    values=torch.tensor([100*t+10*s for t in range(8) for s in range(8)],dtype=torch.float32).reshape(1,64,1,1)
    temporal=summarize_source_values(values,(2,4),'temporal8');spatial=summarize_source_values(values,(2,4),'spatial2x4')
    assert temporal.shape==spatial.shape==(1,8,1,1)
    assert torch.equal(temporal.flatten(),torch.arange(8.)*10+350)
    assert torch.equal(spatial.flatten(),torch.arange(8.)*100+35)
    for kind,summary in [('temporal8',temporal),('spatial2x4',spatial)]:
        target=torch.empty_like(values);fill_grouped_source_values(target,summary,(2,4),kind)
        expected=torch.tensor([350+10*s if kind=='temporal8' else 100*t+35 for t in range(8) for s in range(8)]).reshape_as(values)
        assert torch.equal(target,expected)


def test_projection_preserves_surrounding_read_buffer_and_raw_source():
    raw=torch.randn(1,64,2,4);original=raw.clone();read=torch.full((1,80,2,4),123.)
    summary=summarize_source_values(raw,(2,4),'temporal8');fill_grouped_source_values(read[:,8:72],summary,(2,4),'temporal8')
    assert torch.equal(raw,original) and (read[:,:8]==123).all() and (read[:,72:]==123).all()
