import numpy as np

from scripts.analyze_native_state_region import red_cells


def test_largest_red_component_is_not_green_or_isolated_red_cell():
    rgb=np.zeros((64,64,3),dtype=np.uint8);rgb[:]=[40,160,40]
    rgb[32:64,32:64]=[180,20,50];rgb[:16,:16]=[200,10,20]
    fraction,red,pile=red_cells(rgb,(4,4))
    assert red.sum()==5 and pile.sum()==4 and not pile[0,0]
    assert fraction[3,3]==1 and not red[0,2]
