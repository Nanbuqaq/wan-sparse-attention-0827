import torch
import pytest
from adapters.longlive_sparse.ar_routing import route_history
from scripts.evaluate_complete_attention_capture import construct_routes, evaluate, validate_capture


def capture():
    generator=torch.Generator().manual_seed(81)
    def tensor(tokens):
        return torch.randn(1,tokens,2,16,generator=generator)
    q,k,v,ek,ev=tensor(64),tensor(256),tensor(256),tensor(64),tensor(64)
    frames=torch.tensor([1,2]).repeat_interleave(128).view(1,1,-1).expand(1,2,-1)
    tokens=torch.arange(128).repeat(2).view(1,1,-1).expand(1,2,-1)
    plan=route_history(q,k,frames,tokens,method='rag_dense',density=1.,exact_k_tokens=64)
    return {'scope':'offline_only_complete_attention_post_rope','contains_sink_current_recent':True,
            'teacher_used_by_selector':False,'query':q,'query_unrotated':q.flip(-1),
            'key':k,'key_unrotated':k.flip(-1),'value':v,'exact_key':ek,'exact_value':ev,
            'frame_ids':frames,'token_ids':tokens,'spatial_height':8,'spatial_width':16,
            'route_plan':plan.state_dict()}


def test_complete_teacher_and_actual_byte_controls():
    result=evaluate(capture())
    assert result['records']['captured_route']['output_error']['max_abs']<1e-6
    for name in ('peak_value','count_uniform'):
        method=result['records'][name]
        baseline=result['records']['legacy_matched_'+name]
        for field in ('actual_tokens_per_head','unique_payload_bytes','padded_compact_bytes'):
            assert method[field]==baseline[field]
    assert result['exact_tokens']==64
    assert not result['formal_promotion_allowed']


def test_post_rope_teacher_tensors_cannot_change_online_routes():
    data=capture()
    before=construct_routes(data)
    data['query']=data['query']*100
    data['key']=data['key']*-100
    data['exact_value']=data['exact_value']*100
    after=construct_routes(data)
    assert {key:route.digest() for key,route in before.items()}=={key:route.digest() for key,route in after.items()}
    del data['exact_key']
    with pytest.raises(ValueError,match='complete'):
        validate_capture(data)
