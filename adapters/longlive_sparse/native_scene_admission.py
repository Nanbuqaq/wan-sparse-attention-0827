"""Small cue-gated causal baseline, not a general entity tracker or novel router."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SceneDescriptor:
    archive_version: int
    source_end: int
    source_phase: float
    condition_prototype: torch.Tensor


def has_revisit_cue(current_text):
    text=current_text.removeprefix('The scene transitions. ').strip().lower()
    return text.startswith(('back to ','return to ','returning to ','the camera returns to ','the scene returns to '))


def choose_scene(current_text,current_prototype,candidates,current_frame,*,margin=.05,minimum_cosine=.8,min_gap=32):
    """Selector receives descriptors only; no Q/K/V, Dense output or future text."""
    if not has_revisit_cue(current_text):return dict(selected_version=None,reason='no_explicit_revisit_cue',scores=[])
    if margin<0 or min_gap<0:raise ValueError('invalid frozen selection rule')
    if any(c.source_end>current_frame for c in candidates):raise ValueError('future candidate is forbidden')
    eligible=[c for c in candidates if current_frame-c.source_end>=min_gap]
    if not eligible:return dict(selected_version=None,reason='no_sufficiently_old_scene',scores=[])
    if current_prototype.device.type!='cpu' or any(c.condition_prototype.device.type!='cpu' for c in eligible):
        raise ValueError('selector consumes CPU summaries only')
    q=torch.nn.functional.normalize(current_prototype.float(),dim=0)
    scores=[float(torch.dot(q,torch.nn.functional.normalize(c.condition_prototype.float(),dim=0))) for c in eligible]
    if not all(torch.isfinite(torch.tensor(scores))):raise ValueError('nonfinite condition similarity')
    audit=[dict(archive_version=c.archive_version,source_end=c.source_end,cosine=s) for c,s in zip(eligible,scores)]
    if max(scores)<minimum_cosine:return dict(selected_version=None,reason='below_similarity_floor',scores=audit)
    near=[c for c,s in zip(eligible,scores) if s>=max(minimum_cosine,max(scores)-margin)]
    chosen=max(near,key=lambda c:(c.source_end,c.archive_version))
    return dict(selected_version=chosen.archive_version,reason='latest_within_frozen_similarity_margin',scores=audit)
