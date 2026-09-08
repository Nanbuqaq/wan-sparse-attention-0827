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


def test_metadata_never_reads_dynamic_weight_property():
    layer=torch.nn.Linear(7,11)
    original_getattr=torch.nn.Module.__getattr__
    def dangerous_getattr(self,name):
        if name=='weight':raise AssertionError('metadata triggered parameter materialization')
        return original_getattr(self,name)
    layer.__class__=type('DynamicSwap_Linear',(torch.nn.Linear,),{'__getattr__':dangerous_getattr})
    meter=OperatorWorkMeter()
    meter.observe('text.blocks.0.attn.q',layer,(torch.empty(1,5,7),),{},torch.empty(1,5,11))
    assert next(iter(meter.records.values()))['FLOPs']==2*5*7*11


def test_dynamic_attention_and_cross_scope():
    cls=type('DynamicSwap_T5Attention',(torch.nn.Module,),{})
    module=cls();module.num_heads=2;module.head_dim=4
    meter=OperatorWorkMeter();x=torch.empty(1,5,8)
    meter.observe('text.blocks.0.attn',module,(x,),{},x)
    row=next(iter(meter.records.values()))
    assert row['FLOPs']==4*2*5*5*4 and row['softmax_pairs']==50
    assert stage_for_path('generator.model.blocks.0.cross_attn')=='transformer.cross_attn'
