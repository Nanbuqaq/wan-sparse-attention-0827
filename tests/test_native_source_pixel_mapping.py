import numpy as np
import pytest
from scripts.probe_native_source_foreground import source_token_masks


def test_regular_source_window_preserves_existing_four_pixel_mapping():
    a=np.zeros((32,32,32),dtype=np.bool_);a[3,0,0]=True;a[4,1,1]=True
    assert source_token_masks(a,40)[:,0,0].tolist()==[True,True,False,False,False,False,False,False]


def test_first_latent_has_one_pixel_frame_without_fake_padding():
    a=np.zeros((29,32,32),dtype=np.bool_);a[0,0,0]=True;a[4,0,0]=True;a[28,0,0]=True
    assert source_token_masks(a,0)[:,0,0].tolist()==[True,True,False,False,False,False,False,True]
    with pytest.raises(ValueError):source_token_masks(a,40)
