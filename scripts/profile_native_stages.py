#!/usr/bin/env python3
"""Observe native self/cross Attention and enclosing module CPU launch scopes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--profile-record',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    args,remaining=p.parse_known_args()
    sys.path.insert(0,str(args.source))
    import wan_5b.modules.causal_model as native
    from pipeline import CausalDiffusionInferencePipeline
    original_init=CausalDiffusionInferencePipeline.__init__
    originals={name:getattr(native,name) for name in ('attention','flash_attention')}
    counts=Counter();handles=[]
    def wrap(name,label):
        def call(*a,**kw):
            counts[label]+=1
            with torch.cuda.nvtx.range(label):return originals[name](*a,**kw)
        return call
    def initialize(pipe,*a,**kw):
        original_init(pipe,*a,**kw)
        for i,block in enumerate(pipe._dit_model.blocks):
            for attr in ('self_attn','cross_attn','ffn'):
                module=getattr(block,attr,None)
                if not isinstance(module,torch.nn.Module):continue
                label=f'native/{attr}/layer{i}'
                def before(owner,values,label=label):
                    counts[label]+=1;torch.cuda.nvtx.range_push(label)
                def after(owner,values,result):torch.cuda.nvtx.range_pop()
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after,always_call=True))
    CausalDiffusionInferencePipeline.__init__=initialize
    native.attention=wrap('attention','native/self_attention_core')
    native.flash_attention=wrap('flash_attention','native/cross_attention_core')
    old_argv=sys.argv
    try:
        sys.argv=[str(ROOT/'scripts/run_longlive2_native_reference.py'),'--source',str(args.source),
                  '--generation-profile',*remaining]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        sys.argv=old_argv
        CausalDiffusionInferencePipeline.__init__=original_init
        for name,value in originals.items():setattr(native,name,value)
        for h in handles:h.remove()
        args.profile_record.parent.mkdir(parents=True,exist_ok=True)
        with args.profile_record.open('x') as handle:
            json.dump(dict(scope='observer_only_cudaProfilerApi_generation',
                counts=dict(counts),no_arithmetic_changes=True,
                GPU_attribution_requires_correlated_launches=True,
                timing_is_profiled_not_performance_repetition=True),handle,indent=2)


if __name__=='__main__':main()
