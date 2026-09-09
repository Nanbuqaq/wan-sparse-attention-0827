#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);reference=json.loads((args.control/'summary.json').read_text());base=torch.load(args.control/'latents.pt',map_location='cpu',weights_only=True);rows=[]
    for mode,delta,first in (('raw_reveal',48,8),('raw_away',24,24)):
        root=args.root/mode;d=json.loads((root/'summary.json').read_text());m=d['episode_memory'];plan=m['installation']['admission_plan']
        assert d['status']=='pass' and d['episode_position_policy']=='recent_virtual'
        assert d['noise_sha256']==reference['noise_sha256'] and torch.equal(torch.load(root/'latents.pt',map_location='cpu',weights_only=True)[:,:48],base[:,:48])
        assert plan['source_frames']==list(range(first,first+8)) and plan['virtual_source_frames']==list(range(40,48))
        assert plan['temporal_delta']==delta and plan['spatial_key_channels_unchanged'] and plan['value_unchanged']
        assert m['installation']['cache_metadata_unchanged'] and m['ledger']['demand_H2D_payload_bytes']==377487360
        assert not m['K_positions_preserved_not_rebased'] and m['actual_history_frame_IDs_preserved']
        rows.append(dict(mode=mode,status='pass',pre48_actual_latent_exact=True,delta=delta,ledger=m['ledger']))
    with args.output.open('x') as handle:json.dump(dict(status='pass',cases=rows,quality_not_evaluated=True),handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status='pass',cases=2,quality_not_evaluated=True)))


if __name__=='__main__':main()
