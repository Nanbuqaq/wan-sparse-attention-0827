#!/usr/bin/env python3
"""Numerical guard for bounded partial merge: tails, empty tiles and visibility."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.dataflow_reference import DataflowInputs
from adapters.longlive_sparse.bounded_dataflow_reference import BoundedKVOut


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(60912);rows=[]
    for reuse,empty in [(3,False),(2,False),(1,True)]:
        q=torch.randn(2,65,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(2,321,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        tags=torch.zeros(321,device='cuda',dtype=torch.int32) if empty else torch.arange(321,device='cuda',dtype=torch.int32)%3
        groups=torch.arange(65,device='cuda')*3//65
        mask=torch.ones(65,321,device='cuda',dtype=torch.bool) if reuse==3 else ((groups[:,None]==tags[None])|((groups[:,None]==(tags[None]+1)%3)&(reuse==2)))
        score=(q.float()@k.float().transpose(-1,-2))/128**.5
        expected=score.masked_fill(~mask,-torch.inf).softmax(-1).nan_to_num()@v.float()
        inputs=DataflowInputs(q,k,v,tags,reuse)
        for tiles in (4,8):
            state=BoundedKVOut(inputs,tiles);actual=state.run().clone();again=state.run()
            relative=float((actual.float()-expected).norm()/expected.norm());maximum=float((actual.float()-expected).abs().max())
            assert torch.isfinite(actual).all() and relative<.02 and maximum<.03,(relative,maximum)
            assert torch.equal(actual,again),'running state was not reset on reuse'
            if empty:assert torch.count_nonzero(actual[:,~mask.any(1)])==0
            rows.append(dict(reuse=reuse,empty_query_groups=empty,R=tiles,relative_L2=relative,max_abs=maximum,workspace_bytes=state.workspace_bytes))
    report=dict(status='pass',GPU=torch.cuda.get_device_name(),rows=rows,q_shape=[2,65,128],k_shape=[2,321,128],scope='numerical guard only; tails and fully empty rows, not performance')
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
