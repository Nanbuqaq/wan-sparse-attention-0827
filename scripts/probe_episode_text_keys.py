#!/usr/bin/env python3
"""Probe current-instruction/past-episode keys using the existing FP32 UMT5.

No video, future output or image teacher enters this probe. All candidate text
scoring at event i is restricted to prior episodes0..i-1. This is not yet a
promoted automatic retrieval method.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required for the unchanged text encoder')
    from adapters.longlive_sparse.upstreams import configure_upstream_paths
    configure_upstream_paths()
    from utils.wan_wrapper import WanTextEncoder
    from utils.memory import DynamicSwapInstaller
    begin=time.perf_counter();encoder=WanTextEncoder().eval();DynamicSwapInstaller.install_model(encoder,device=torch.device('cuda'))
    load_s=time.perf_counter()-begin
    spec_path=ROOT/'configs/system/memory_revisit_development.json';spec=json.loads(spec_path.read_text())
    records=[]
    for scenario in spec['scenarios']:
        embeddings=[];encode_times=[]
        for segment in scenario['segments']:
            start=time.perf_counter();values=encoder(text_prompts=[segment['prompt']])['prompt_embeds']
            torch.cuda.synchronize();encode_times.append(time.perf_counter()-start)
            embeddings.append(values.cpu())
        torch.save(embeddings,args.output/(scenario['id']+'_prompt_embeds.pt'))
        pooled=[]
        for emb in embeddings:
            valid=emb.abs().sum(-1)>0
            pooled.append((emb.float()*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[...,None])
        means=F.normalize(torch.cat(pooled),dim=-1)
        similarity=means@means.T
        decisions=[]
        for current in range(1,len(embeddings)):
            cosine=similarity[current,:current]
            contrast=cosine-similarity[current-1,:current]
            decisions.append(dict(event=current,current_role=scenario['segments'][current]['role'],
                prior_roles=[s['role'] for s in scenario['segments'][:current]],
                cosine_scores=cosine.tolist(),contrast_to_previous_scores=contrast.tolist(),
                cosine_choice=int(cosine.argmax()),contrast_choice=int(contrast.argmax()),
                future_episode_keys_consulted=False))
        records.append(dict(scenario=scenario['id'],segments=scenario['segments'],decisions=decisions,
            known_input_cosine_matrix=similarity.tolist(),encode_times_s=encode_times,
            embedding_dtype=str(embeddings[0].dtype)))
        print(json.dumps(records[-1]),flush=True)
    result=dict(status='pass',gpu=torch.cuda.get_device_name(),encoder_load_s=load_s,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),records=records,
        uses_existing_generator_condition_encoder=True,video_output_access=False,
        automatic_retrieval_quality_claim=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
