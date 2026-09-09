from pathlib import Path
from types import SimpleNamespace

import torch

from adapters.longlive_sparse.native_cut_ablation import NativeRopePhaseFreeze,condition_aliases
from scripts.run_longlive2_native_reference import CachedNativeTextEncoder,native_cut_schedule


def test_alias_changes_only_late_prefixes_and_preserves_raw_metadata_list():
    root=Path(__file__).resolve().parents[1]
    _,ps=native_cut_schedule(root,'settled_bead_visible_control');raw=list(ps[0]);aliases=condition_aliases(raw)
    assert raw[2] not in aliases and raw[6] in aliases and raw[12] in aliases
    assert raw[6].startswith('The scene transitions. ')
    values={key:torch.full((1,2,3),float(i)) for i,key in enumerate(dict.fromkeys(raw))}
    encoder=CachedNativeTextEncoder(values,'cpu',aliases)
    out=encoder(text_prompts=raw)['prompt_embeds']
    assert torch.equal(out[2],values[raw[2]][0])
    assert torch.equal(out[6],values[raw[6].removeprefix('The scene transitions. ')][0])


def test_phase_freeze_does_not_change_earlier_phase_and_overrides_each_later_cut():
    p=SimpleNamespace(frame_seq_length=880,_dit_model=SimpleNamespace(rope_temporal_offset=8))
    observer=NativeRopePhaseFreeze(p)
    observer.before(None,(),{'current_start':40*880})
    assert p._dit_model.rope_temporal_offset==8
    p._dit_model.rope_temporal_offset=16;observer.before(None,(),{'current_start':48*880})
    assert p._dit_model.rope_temporal_offset==8
    p._dit_model.rope_temporal_offset=24;observer.before(None,(),{'current_start':96*880})
    assert p._dit_model.rope_temporal_offset==8
    assert observer.audit()['events']==[{'frame':48,'before':16.,'used':8.},{'frame':96,'before':24.,'used':8.}]
