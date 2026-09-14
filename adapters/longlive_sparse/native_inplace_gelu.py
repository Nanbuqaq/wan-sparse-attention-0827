"""Inference-only output reuse for audited Linear-GELU-Linear FFNs.

Uses the same ATen GELU operator and approximation as nn.GELU. Only the private
first Linear output is overwritten; residual inputs, KV and parameters are not.
"""
import torch


class InplaceNativeGELU(torch.nn.Module):
    def __init__(self,approximate):
        super().__init__();self.approximate=approximate;self.calls=0;self.maximum_input_bytes=0

    def forward(self,x):
        if self.training or torch.is_grad_enabled():raise RuntimeError('in-place GELU is inference-only')
        if not x.is_contiguous():raise RuntimeError('audited Linear output must be contiguous')
        self.calls+=1;self.maximum_input_bytes=max(self.maximum_input_bytes,x.numel()*x.element_size())
        return torch.ops.aten.gelu.out(x,approximate=self.approximate,out=x)


class NativeInplaceGelu:
    def __init__(self,model,expected_layers=30):
        if model.training:raise ValueError('set the generator to eval before changing inference buffers')
        if len(model.blocks)!=expected_layers:raise ValueError('unexpected transformer block count')
        replacements=[]
        for block in model.blocks:
            seq=block.ffn
            if (type(seq) is not torch.nn.Sequential or len(seq)!=3 or type(seq[0]) is not torch.nn.Linear
                or type(seq[1]) is not torch.nn.GELU or type(seq[2]) is not torch.nn.Linear
                or seq[0].out_features!=seq[2].in_features):
                raise ValueError('only the audited Linear-GELU-Linear topology is supported')
            if any(m._forward_hooks or m._forward_pre_hooks for m in (seq[0],seq[1])):
                raise ValueError('activation observers may retain an alias of the private Linear output')
            replacements.append((seq,InplaceNativeGELU(seq[1].approximate).eval()))
        for seq,replacement in replacements:seq[1]=replacement
        self.layers=[x[1] for x in replacements]

    def audit(self):
        return dict(enabled=True,patched_FFNs=len(self.layers),calls=sum(x.calls for x in self.layers),
            maximum_single_activation_output_bytes_avoided=max(x.maximum_input_bytes for x in self.layers),
            operator='aten.gelu.out with out=input',approximation=sorted({x.approximate for x in self.layers}),
            scope='private first-Linear output in inference-only DiT FFNs; native operator unchanged',
            actual_peak_saving_requires_measurement=True,KV_parameters_and_residual_inputs_unchanged=True)
