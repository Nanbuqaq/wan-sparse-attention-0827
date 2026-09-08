#!/usr/bin/env python3
"""Close allocation and scene-reset gates, preserving a failed compiler comparison."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);records=[]
    def read(path):
        d=json.loads(path.read_text());records.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=d['status']));return d
    capacity=[]
    for arm,lane,baseline in (('positive32',0,'none'),('positive128',1,'window128')):
        new=read(args.root/'capacity_gate64_v1'/arm/'summary.json')
        old=read(args.root/'gate64_v1'/f'lane{lane}'/baseline/'summary.json')
        assert new['status']=='pass' and new['observer_noise_latent_RGB_equivalence']
        assert new['latent_sha256']==old['latent_sha256'] and new['pixels']['raw_RGB_sha256']==old['pixels']['raw_RGB_sha256']
        assert new['native_positive_and_negative_KV_bytes']*2==old['native_positive_and_negative_KV_bytes']
        capacity.append(dict(arm=arm,full_gate_latent_RGB_exact=True,
            old_KV_bytes=old['native_positive_and_negative_KV_bytes'],new_KV_bytes=new['native_positive_and_negative_KV_bytes'],
            old_generation_peak=old['generation_peak_allocated_bytes'],new_generation_peak=new['generation_peak_allocated_bytes']))
    failed=read(args.root/'capacity_gate64_v1/positive128_triton331/summary.json')
    assert failed['status']=='fail' and failed['triton_version']=='3.3.1'
    a=torch.load(args.root/'capacity_gate64_v1/positive128/latents.pt',map_location='cpu',weights_only=True).float()
    b=torch.load(args.root/'capacity_gate64_v1/positive128_triton331/latents.pt',map_location='cpu',weights_only=True).float()
    compiler=dict(status='fail',cross_compiler_exact_gate=False,promotion_allowed=False,
        relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)),max_abs=float((a-b).abs().max()),
        per_chunk_relative_l2=[float(torch.linalg.vector_norm(a[:,i:i+8]-b[:,i:i+8])/torch.linalg.vector_norm(a[:,i:i+8])) for i in range(0,64,8)],
        classification='generated_but_failed_complete_trajectory_equivalence_not_proven_roundoff')
    base=read(args.root/'capacity_gate64_v1/positive32/summary.json')
    base_latent=torch.load(args.root/'capacity_gate64_v1/positive32/latents.pt',map_location='cpu',weights_only=True)
    contexts=[]
    for mode,source in (('raw_reveal',8),('raw_away',24)):
        root=args.root/'context_gate64_v1'/mode;d=read(root/'summary.json');m=d['episode_memory']
        latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
        assert d['status']=='pass' and d['triton_version']=='3.2.0' and d['noise_sha256']==base['noise_sha256']
        assert torch.equal(latent[:,:48],base_latent[:,:48])
        assert m['capture']['source_frames']==list(range(source,source+8))
        assert m['observed_return_attention_shapes']==[dict(query_start_latent=48,Q=1024,K=3072,calls=150),dict(query_start_latent=56,Q=1024,K=4096,calls=150)]
        transition=m['installation']['cache_metadata_transition'];before=transition['before'];after=transition['after']
        assert after==dict(before,local_end_index=2048) and before['local_end_index']==4096
        contexts.append(dict(mode=mode,status='pass',actual_pre48_latent_exact=True,
            recall_bytes=m['ledger']['demand_H2D_payload_bytes'],actual_return_shapes=m['observed_return_attention_shapes']))
    assert contexts[0]['recall_bytes']==contexts[1]['recall_bytes']==377487360
    result=dict(status='complete',capacity_gates=capacity,context_gates=contexts,compiler_gate=compiler,
        semantic_quality_conclusion=False,source_records=records,missing=0)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status=result['status'],capacity_pass=2,context_pass=2,preserved_compiler_fail=1,missing=0)))


if __name__=='__main__':main()
