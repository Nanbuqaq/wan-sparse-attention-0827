"""Synthetic GPU mutation-race gate; not a video or long-history result."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.staged_scene_archive import StagedSceneArchive


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);rows=[]
    for protected in (False,True):
        ft=128;frame=8 if protected else 32;n=32*ft;caches=[]
        for layer in range(4):
            caches.append(dict(k=torch.full((1,n,24,128),layer+1.,device='cuda',dtype=torch.bfloat16),
                v=torch.full((1,n,24,128),10*(layer+1.),device='cuda',dtype=torch.bfloat16),
                local_end_index=torch.tensor(frame*ft,device='cuda'),global_end_index=torch.tensor(frame*ft,device='cuda')))
        pipe=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,num_frame_per_block=8,
            local_attn_size=32,sink_size=8,frame_seq_length=ft,kv_cache_pos=caches)
        archive=StagedSceneArchive(pipe,archive_budget=256*1024**2)
        archive.last_commit=dict(end=frame,phase=0.,prototype=torch.ones(4))
        archive._archive_last_scene(frame)
        for layer,c in enumerate(caches):
            archive.before_layer_mutation(layer)
            start=8*ft if protected else 0
            c['k'][:,start:].fill_(-99);c['v'][:,start:].fill_(-199)
        archive.wait_for_bank(archive.banks[0]);archive.close();torch.cuda.synchronize()
        correct=[]
        for layer,(k,v) in enumerate(archive.banks[0]['kv']):
            correct.append(bool((k==layer+1).all() and (v==10*(layer+1)).all()))
        assert all(correct)
        assert not archive.active_job.source and not archive.active_job.destination
        result=archive.audit();rows.append(dict(protected=protected,correct_layers=correct,audit=result))
    (a.output/'result.json').write_text(json.dumps(dict(status='pass',GPU=torch.cuda.get_device_name(),rows=rows,
        synthetic_race_gate_only=True,full_video_equivalence_not_established=True),indent=2))


if __name__=='__main__':main()
