import torch
from adapters.longlive_sparse.native_causal_block_memory import source_head_scores
from adapters.longlive_sparse.native_source_context_proxy import source_context_scores
from adapters.longlive_sparse.native_source_context_proxy import kept_context_means


def test_kept_context_can_suppress_irrelevant_queries_before_score_aggregation():
    q=torch.tensor([[[1.]], [[-1.]]]);sk=torch.tensor([[[2.]], [[-2.]]]);sv=torch.tensor([[[2.]], [[1.]]])
    counts=torch.ones(2);kk=torch.tensor([[[20.]]]);kv=torch.zeros_like(kk)
    local=source_head_scores(q,sk,sv,counts,'mass_value')
    joint=source_context_scores(q,sk,sv,counts,kk,kv,torch.ones(1),'mass_value')
    assert local.argmax(-1).item()==0
    assert joint.argmax(-1).item()==1


def test_contrast_reference_includes_kept_values_and_group_multiplicity():
    q=torch.zeros(1,1,1);sk=torch.zeros(2,1,1);sv=torch.tensor([[[1.]], [[-1.]]])
    kk=torch.zeros(1,1,1);kv=torch.tensor([[[10.]]])
    # At equal logits, weights are1/4,1/4,2/4 and the joint mean is5.
    score=source_context_scores(q,sk,sv,torch.ones(2),kk,kv,torch.tensor([2.]),'contrast_value')
    assert torch.allclose(score,torch.tensor([[1.,1.5]]))


def test_kept_summary_excludes_source_slots_and_matches_frame_tail_means():
    k=torch.arange(4*80*2*3).reshape(1,4*80,2,3).bfloat16();v=k+10
    km,vm,counts=kept_context_means(k,v,source_start=80,source_tokens=80,frame_tokens=80)
    assert counts.tolist()==[64,16]*3
    for actual,raw in ((km,k),(vm,v)):
        expected=torch.stack([raw[0,frame*80+a:frame*80+b].float().mean(0) for frame in (0,2,3) for a,b in ((0,64),(64,80))])
        assert torch.equal(actual,expected)
