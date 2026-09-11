"""Source-locked SAM2 return adapter; no changes to prediction or mask thresholds."""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

PREDICTOR_SHA = 'f13e5f9d94e5c8d9d2c3622dab20c8f334c089ef2ee5ea8e199da7d332b029ba'


def verify_predictor_source(predictor):
    path = Path(inspect.getfile(type(predictor)))
    if hashlib.sha256(path.read_bytes()).hexdigest() != PREDICTOR_SHA:
        raise ValueError('compact return requires the qualified SAM2 predictor source')


@torch.inference_mode()
def predict_selected_mask(predictor, box):
    if not predictor._is_image_set:
        raise RuntimeError('source image must be set before mask prediction')
    mask_input, coords, labels, transformed_box = predictor._prep_prompts(
        None, None, np.asarray(box, dtype=np.float32), None, True)
    masks, scores, _ = predictor._predict(
        coords, labels, transformed_box, mask_input, True, return_logits=False)
    if masks.shape[0] != 1 or masks.dtype != torch.bool:
        raise ValueError('one source image with thresholded boolean masks required')
    # Preserve public float32 scores and NumPy's first-maximum tie convention.
    scores_cpu = scores.squeeze(0).float().detach().cpu().numpy()
    if not np.isfinite(scores_cpu).all():
        raise ValueError('nonfinite source mask quality scores')
    chosen = int(np.argmax(scores_cpu))
    selected = masks[0, chosen].detach().cpu().numpy()
    return selected, scores_cpu, chosen
