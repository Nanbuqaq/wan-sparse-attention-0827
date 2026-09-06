import numpy as np
from scripts.build_sam2_oracle_masks import select_mask


def test_automatic_teacher_rule_rejects_full_frame_and_prefers_largest_central():
    mask=np.ones((10,10),dtype=bool)
    candidates=[{'area':100,'bbox':[0,0,10,10],'segmentation':mask,'predicted_iou':1.},
                {'area':20,'bbox':[3,3,4,5],'segmentation':mask,'predicted_iou':.9},
                {'area':30,'bbox':[2,2,5,6],'segmentation':mask,'predicted_iou':.8}]
    assert select_mask(candidates,10,10)==2
    assert select_mask(candidates[:1],10,10) is None
