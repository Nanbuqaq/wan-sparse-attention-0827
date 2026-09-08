"""Shape-derived arithmetic accounting, never hardware instruction counters.

Only hooks module metadata (no tensor contents/CPU copies/synchronization).
Linear leaves include unmerged LoRA A/B; composite modules are not double-counted.
Convolution uses direct dense-equivalent FLOPs, regardless of cuDNN algorithm.
"""
from collections import defaultdict
import math

import torch


def linear_flops(input_shape,in_features,out_features):
    if input_shape[-1]!=in_features:raise ValueError('linear input width mismatch')
    return 2*math.prod(input_shape[:-1])*in_features*out_features


def convolution_flops(input_shape,output_shape,kernel_size,groups,transposed=False):
    if groups<1 or input_shape[1]%groups or output_shape[1]%groups:
        raise ValueError('invalid convolution groups')
    return (2*math.prod(input_shape)*output_shape[1]//groups*math.prod(kernel_size) if transposed else
            2*math.prod(output_shape)*input_shape[1]//groups*math.prod(kernel_size))


def stage_for_path(path):
    if path.startswith('text.'):return 'text.encode_with_dynamic_swap'
    if path.startswith('vae.'):return 'vae.decode_complete'
    if '.self_attn.' in path:
        for component in ('q','k','v','o'):
            if f'.self_attn.{component}.' in path or path.endswith('.self_attn.'+component):
                return 'self_attention.'+component
        return 'transformer.self_attn'
    for fragment,stage in (('.cross_attn.','transformer.cross_attn'),('.ffn.','transformer.ffn'),
            ('.patch_embedding','generator.patch_embedding'),('.time_embedding','generator.time_embedding'),
            ('.time_projection','generator.time_projection'),('.text_embedding','generator.text_embedding'),('.head.','generator.head')):
        if fragment in path:return stage
    return 'generator.other_modules'


class OperatorWorkMeter:
    def __init__(self):
        self.records={};self.handles=[];self.roots=[]

    def attach(self,pipeline):
        seen=set()
        for prefix,root in (('generator',pipeline.generator),('text',pipeline.text_encoder),('vae',pipeline.vae)):
            self.roots.append(dict(name=prefix,parameters=sum(p.numel() for p in root.parameters()),
                parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in root.parameters())))
            for name,module in root.named_modules():
                if id(module) in seen:continue
                path=prefix+'.'+name
                supported=isinstance(module,(torch.nn.Linear,torch.nn.modules.conv._ConvNd))
                special=(module.__class__.__name__=='T5Attention' and prefix=='text') or (module.__class__.__name__=='AttentionBlock' and prefix=='vae')
                cross=(path.endswith('.cross_attn') and 'T2VCrossAttention' in module.__class__.__name__)
                if supported or special or cross:
                    seen.add(id(module))
                    self.handles.append(module.register_forward_hook(
                        lambda m,a,k,o,path=path:self.observe(path,m,a,k,o),with_kwargs=True))

    def observe(self,path,module,args,kwargs,output):
        x=args[0] if args else kwargs.get('x')
        if not isinstance(x,torch.Tensor) or not isinstance(output,torch.Tensor):
            raise ValueError('audited module signature changed')
        xs,ys=tuple(x.shape),tuple(output.shape);pairs=0
        if isinstance(module,torch.nn.Linear):
            kind='linear';flops=linear_flops(xs,module.in_features,module.out_features)
        elif isinstance(module,torch.nn.modules.conv._ConvNd):
            kind='convolution_direct_equivalent'
            flops=convolution_flops(xs,ys,module.kernel_size,module.groups,module.transposed)
        elif module.__class__.__name__=='T5Attention':
            context=(args[1] if len(args)>1 else kwargs.get('context'))
            context=x if context is None else context
            pairs=xs[0]*module.num_heads*xs[1]*context.shape[1]
            flops=4*pairs*module.head_dim;kind='T5_attention_core'
        elif path.startswith('vae.') and module.__class__.__name__=='AttentionBlock':
            b,c,t,h,w=xs;pairs=b*t*(h*w)**2;flops=4*pairs*c;kind='VAE_attention_core'
        else:
            context=args[1] if len(args)>1 else kwargs.get('context')
            if context is None:raise ValueError('cross attention context is absent')
            pairs=xs[0]*module.num_heads*xs[1]*context.shape[1]
            flops=4*pairs*(module.dim//module.num_heads);kind='text_cross_attention_core'
        key=(path,kind,xs,ys,str(x.dtype),str(output.dtype))
        if key not in self.records:
            weight=getattr(module,'weight',None)
            weight_bytes=weight.numel()*weight.element_size() if isinstance(weight,torch.Tensor) else 0
            self.records[key]=dict(module=path,stage=stage_for_path(path),kind=kind,input_shape=xs,output_shape=ys,
                input_dtype=str(x.dtype),output_dtype=str(output.dtype),device=str(x.device),
                weight_dtype=str(weight.dtype) if isinstance(weight,torch.Tensor) else None,
                unmerged_lora_leaf=('.lora_A.' in path or '.lora_B.' in path),calls=0,FLOPs=0,softmax_pairs=0,
                operand_once_tensor_bytes_per_call=(x.numel()*x.element_size()+output.numel()*output.element_size()+weight_bytes
                    if kind in ('linear','convolution_direct_equivalent') else None))
        row=self.records[key];row['calls']+=1;row['FLOPs']+=int(flops);row['softmax_pairs']+=int(pairs)

    def detach(self):
        for h in self.handles:h.remove()
        self.handles=[]

    def result(self,stats,head_dim):
        stages=defaultdict(lambda:dict(FLOPs=0,unmerged_lora_FLOPs=0,softmax_pairs=0,module_calls=0,operand_once_tensor_bytes=0))
        for row in self.records.values():
            out=stages[row['stage']];out['FLOPs']+=row['FLOPs'];out['softmax_pairs']+=row['softmax_pairs'];out['module_calls']+=row['calls']
            if row['unmerged_lora_leaf']:out['unmerged_lora_FLOPs']+=row['FLOPs']
            if row['operand_once_tensor_bytes_per_call'] is not None:
                out['operand_once_tensor_bytes']+=row['calls']*row['operand_once_tensor_bytes_per_call']
        attention_pairs=int(stats['executed_qk_pairs'])
        stages['history_and_exact_attention_core'].update(FLOPs=4*head_dim*attention_pairs,softmax_pairs=attention_pairs,
            module_calls=stats['calls'],operand_once_tensor_bytes=None)
        return dict(schema='shape_derived_operator_work_v1',stages=dict(stages),module_shape_records=list(self.records.values()),
            model_inventory=self.roots,head_dim=head_dim,attention_plan_pairs=attention_pairs,
            FMA_counted_as_two_FLOPs=True,LoRA_leaves_counted_separately=True,
            nonlinear_integer_reduction_and_metadata_ops_not_in_GEMM_FLOPs=True,
            convolution_FLOPs_are_direct_equivalent_not_actual_cuDNN_instructions=True,
            attention_FLOPs_are_useful_plan_work_not_padded_MMA_instructions=True,
            operand_once_bytes_are_a_tensor_IO_model_not_measured_HBM=True)
