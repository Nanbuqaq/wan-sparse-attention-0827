from types import SimpleNamespace

import torch

from adapters.longlive_sparse.operator_work import OperatorWorkMeter,linear_flops,convolution_flops,stage_for_path


def test_arithmetic_and_lora_path_scope():
    assert linear_flops((2,5,7),7,11)==2*2*5*7*11
    assert convolution_flops((1,4,5,5),(1,6,3,3),(3,3),2)==2*6*3*3*2*9
    assert stage_for_path('generator.model.blocks.0.self_attn.q.lora_A.default')=='self_attention.q'
    assert stage_for_path('generator.model.blocks.0.ffn.0.base_layer')=='transformer.ffn'


def test_hooks_do_not_change_outputs_and_count_leaf_work():
    model=torch.nn.Sequential(torch.nn.Linear(7,11),torch.nn.GELU(),torch.nn.Linear(11,7))
    pipe=SimpleNamespace(generator=model,text_encoder=torch.nn.Identity(),vae=torch.nn.Identity())
    x=torch.randn(2,5,7);reference=model(x)
    meter=OperatorWorkMeter();meter.attach(pipe);actual=model(x);meter.detach()
    data=meter.result({'executed_qk_pairs':30,'calls':2},4)
    assert torch.equal(reference,actual)
    assert sum(r['FLOPs'] for r in data['module_shape_records'])==2*2*2*5*7*11
    assert data['stages']['history_and_exact_attention_core']['FLOPs']==480
