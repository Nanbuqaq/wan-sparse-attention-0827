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


def test_mutual_geometry_merges_nearest_pair_but_not_a_one_way_background_neighbor():
    rows=[]
    for start,x,y in ((0,100,100),(10,240,100),(20,900,500)):
        rows += [dict(id=start+i,bbox=[x+30*i,y,10,10],spatial_tokens=1,pixel_area=100) for i in range(3)]
    box,ids,groups=component_box(rows,policy='mutual_geometry')
    assert ids==[0,1,2,10,11,12] and box==[68,68,342,142]
    assert groups==[[0,1,2,10,11,12],[20,21,22]]
