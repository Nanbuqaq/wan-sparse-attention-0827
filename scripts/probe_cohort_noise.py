"""Own-platform native5B noise hashes for each registered case seed."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.native_duration_probe import duration_noise


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--length',type=int,choices=(128,728),required=True);a=p.parse_args()
    torch.set_num_threads(2);rows=[]
    for seed in sorted(set(a.seeds)):
        for device in range(0,torch.cuda.device_count(),2):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            shape=(1,a.length,48,44,80)
            x=(torch.randn(*shape,device=f'cuda:{device}',dtype=torch.bfloat16) if a.length==128 else
               duration_noise(shape,base_length=128,seed=seed,device=f'cuda:{device}'))
            x=x.cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode())
            h.update(json.dumps(list(x.shape)).encode());h.update(x.view(torch.uint8).numpy().tobytes())
            rows.append(dict(seed=seed,device=device,GPU=torch.cuda.get_device_name(device),noise_sha256=h.hexdigest()))
        if len({r['noise_sha256'] for r in rows if r['seed']==seed})!=1:raise RuntimeError('same-seed lane noise differs')
    print(json.dumps(dict(shape=list(shape),seeds=sorted(set(a.seeds)),rows=rows)))


if __name__=='__main__':main()
