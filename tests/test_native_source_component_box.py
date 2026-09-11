from scripts.derive_native_source_component_box import component_box
import pytest


def test_large_background_cannot_bridge_separate_part_components():
    rows=[dict(id=i,bbox=[100+30*i,100,10,10],spatial_tokens=1,pixel_area=100) for i in range(3)]
    rows+=[dict(id=9,bbox=[0,0,1000,700],spatial_tokens=500,pixel_area=700000),
           dict(id=10,bbox=[900,500,10,10],spatial_tokens=1,pixel_area=100)]
    box,ids,groups=component_box(rows)
    assert ids==[0,1,2] and box==[68,68,202,142] and groups==[[0,1,2]]
    with pytest.raises(ValueError):component_box(rows[-2:])


def test_all_policy_preserves_separated_eligible_components_without_changing_edges():
    rows=[dict(id=i,bbox=[100+30*i,100,10,10],spatial_tokens=1,pixel_area=100) for i in range(3)]
    rows += [dict(id=10+i,bbox=[700+30*i,500,10,10],spatial_tokens=1,pixel_area=100) for i in range(3)]
    _,largest,groups=component_box(rows)
    box,all_ids,all_groups=component_box(rows,policy='all')
    assert largest==[0,1,2] and groups==all_groups
    assert all_ids==[0,1,2,10,11,12] and box==[68,68,802,542]
