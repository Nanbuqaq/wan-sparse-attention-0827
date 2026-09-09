import numpy as np
import torch

from scripts.analyze_rebound_source_budget import nearby_nonred,energy_indices


def test_nearby_control_preserves_frame_counts_and_excludes_all_red():
    pile=np.zeros((2,5,5),dtype=bool);pile[:,2,2]=True;pile[1,2,3]=True
    red=pile.copy();red[:,1,2]=True
    chosen=nearby_nonred(pile,red,frames=2,height=5,width=5)
    assert len(chosen)==3 and not any(red.flat[i] for i in chosen)
    assert sum(i<25 for i in chosen)==1 and sum(i>=25 for i in chosen)==2


def test_energy_selection_is_past_value_only_and_matches_each_frame_budget():
    v=torch.arange(16,dtype=torch.float32).reshape(8,2,1)
    selected=energy_indices(v,[1,2],frame_tokens=4)
    assert selected==[3,6,7]
