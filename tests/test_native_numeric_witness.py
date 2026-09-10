import torch
from types import SimpleNamespace
from adapters.longlive_sparse.native_numeric_witness import NativeNumericWitness,TARGETS


def test_only_original_failures_and_neighbor_heads_are_registered():
    assert set(TARGETS)=={(96,0,29),(96,4,4),(120,0,29),(120,3,29)}
    assert sum(map(len,TARGETS.values()))==8


def test_untargeted_observation_returns_original_tensor_without_capture():
    probe=NativeNumericWitness(SimpleNamespace(frame_seq_length=4),(2,2))
    probe.active=(0,0);probe.layer=0
    q=torch.zeros(1,8,2,4);k=v=q
    result=torch.ones_like(q)
    assert probe.observe(lambda *a,**kw:result,q,k,v) is result
    assert probe.records==[]
