#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);rows=[]
    def pixel_hash(path):
        h=hashlib.sha256();count=0
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if count==445:break
                h.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
        if count!=445:raise ValueError('truncated prefix')
        return h.hexdigest()
    for mode in ('raw_reveal','raw_away'):
        root=args.root/mode;control=args.control/mode
        try:
            d=json.loads((root/'summary.json').read_text());c=json.loads((control/'summary.json').read_text())
            assert d['status']==c['status']=='pass'
            for key in ('seed','cut_scenario','noise_sha256','latent_shape','native_KV_allocation_policy','triton_version','fixed_native_adaln_recipe'):
                assert d[key]==c[key],key
            a=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
            b=torch.load(control/'latents.pt',map_location='cpu',weights_only=True)
            assert torch.equal(a[:,:112],b[:,:112])
            assert pixel_hash(root/'video.mp4')==pixel_hash(control/'video.mp4')
            m=d['episode_memory'];old=c['episode_memory']
            for field in ('demand_H2D_payload_bytes','source_replication_D2D_bytes','CPU_archive_peak_bytes'):
                assert m['ledger'][field]==old['ledger'][field]==2595225600
            assert m['observed_return_attention_shapes']==old['observed_return_attention_shapes']
            event=m['source_pin_override']
            assert event['completed_latent']==104 and event['KV_data_copied_bytes']==0 and event['metadata_written_bytes']==480
            assert event['after']==dict(event['before'],pinned_start=7040,pinned_len=7040)
            rows.append(dict(mode=mode,status='pass',pre112_latent_and_445_decoded_pixels_exact=True,
                             complete_latent_changed=not torch.equal(a,b),same_attention_shapes_and_KV_payloads=True,event=event))
        except (OSError,ValueError,KeyError,AssertionError) as error:rows.append(dict(mode=mode,status='fail',error=str(error)))
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',cases=rows,expected_cases=2,quality_not_inferred=True)
    with (args.root/'source_pin_terminal.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
