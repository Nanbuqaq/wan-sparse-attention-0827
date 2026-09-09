import pytest

from adapters.longlive_sparse.native_region_layout import region_layout_plan


@pytest.mark.parametrize('layout',['exact','block64','page256','frame','spatial4','spatial8'])
def test_physical_padding_and_runs_do_not_change_selected_coordinates(layout):
    selected=[0,6,7,34,36,69]
    plan=region_layout_plan(selected,layout,frames=2,height=5,width=7)
    packed=[plan['physical_to_logical'][i] for i in plan['packed_physical_indices']]
    assert [packed[i] for i in plan['gather_indices']]==selected
    assert plan['valid_payload_tokens']+plan['padding_tokens']==plan['scheduled_tokens']
    reconstructed=[]
    for start,end,destination in plan['runs']:
        assert destination==len(reconstructed)
        reconstructed.extend(range(start,end))
    assert reconstructed==plan['packed_physical_indices']
    assert plan['logical_coordinate_sha256']==region_layout_plan(selected,'exact',frames=2,height=5,width=7)['logical_coordinate_sha256']


def test_invalid_roi_and_layout_rejected():
    with pytest.raises(ValueError):region_layout_plan([], 'exact')
    with pytest.raises(ValueError):region_layout_plan([7040], 'exact')
    with pytest.raises(ValueError):region_layout_plan([1], 'unknown')
