#!/usr/bin/env python3
"""Offline current-condition/closed-scene text prototype characterization.

Not a video method or trained router. Tests a causal-available signal absent
from first-denoise layer0 Q under fixed noise/geometry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_longlive2_native_reference import native_cut_schedule,episode_source_and_target


def condition_probe_cases(root):
    rows=[]
    for scenario in ('generated_bead_state_cut_revisit','settled_bead_revisit','generated_patchwork_toy_cut_revisit'):
        segments,prompts=native_cut_schedule(root,scenario);source_end,target=episode_source_and_target(segments)
        initial_end=segments[1]['start_latent']
        candidates=[dict(role=role,end=end,text=prompts[0][end//8-1]) for role,end in (
            ('initial',initial_end),('source',source_end),('away',target))]
        queries=[dict(role='return_without_restatement',text=prompts[0][target//8],expected_candidate='source'),
            dict(role='explicit_initial_state',text='The scene transitions. '+candidates[0]['text'].removeprefix('The scene transitions. '),expected_candidate='initial'),
            dict(role='explicit_away',text='The scene transitions. '+candidates[2]['text'].removeprefix('The scene transitions. '),expected_candidate='away')]
        rows.append(dict(scenario=scenario,candidates=candidates,queries=queries))
    return rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('physical GPU required')
    source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    assert source_sha=='6b36d20ec6f7958d29d11a704dfa64611a9f2572'
    assert json.loads((args.assets/'assets_manifest.json').read_text())['status']=='pass'
    cases=condition_probe_cases(ROOT);texts=list(dict.fromkeys(r['text'] for c in cases for r in c['candidates']+c['queries']))
    sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from utils.wan_5b_wrapper import WanTextEncoder
    started=time.perf_counter();model=WanTextEncoder().to(dtype=torch.bfloat16).eval();torch.cuda.synchronize()
    load_s=time.perf_counter()-started;prototypes={};services=[]
    for text in texts:
        began=time.perf_counter();encoded=model(text_prompts=[text])['prompt_embeds'];torch.cuda.synchronize()
        encoded_s=time.perf_counter()-began
        _,mask=model.tokenizer([text],return_mask=True,add_special_tokens=True)
        valid=mask[0].bool().to(encoded.device);tokens=encoded[0,valid].float()
        mean=tokens.mean(0);unit_mean=torch.nn.functional.normalize(tokens,dim=-1).mean(0)
        torch.cuda.synchronize();summarized=time.perf_counter()
        prototype=torch.stack((torch.nn.functional.normalize(mean,dim=0),torch.nn.functional.normalize(unit_mean,dim=0))).cpu()
        assert torch.isfinite(prototype).all();prototypes[text]=prototype
        services.append(dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(),valid_tokens=int(valid.sum()),
            native_encode_wall_s=encoded_s,tokenization_and_summary_s=summarized-began-encoded_s,
            summary_D2H_and_normalize_s=time.perf_counter()-summarized,prototype_bytes=prototype.numel()*prototype.element_size()))
    rows=[]
    for case in cases:
        for query in case['queries']:
            candidates=case['candidates'];matrix=torch.stack([prototypes[c['text']] for c in candidates])
            scores=(matrix*prototypes[query['text']][None]).sum(-1)
            rules=[]
            for index,name in enumerate(('masked_token_mean','mean_of_unit_tokens')):
                values=scores[:,index];best=int(values.argmax())
                # Predeclared characterization rules only, not fitted on video outcomes.
                near=[i for i,v in enumerate(values) if float(v)>=float(values.max())-.05]
                latest=max(near,key=lambda i:candidates[i]['end'])
                rules.append(dict(proxy=name,scores={c['role']:float(v) for c,v in zip(candidates,values)},
                    cosine_argmax=candidates[best]['role'],within_0p05_then_latest=candidates[latest]['role']))
            rows.append(dict(scenario=case['scenario'],query_role=query['role'],query=query['text'],
                expected_candidate_by_constructed_prompt=query['expected_candidate'],candidates=candidates,rules=rules))
    torch.save(dict(prototypes=prototypes,offline_characterization_only=True),args.output/'text_prototypes.pt')
    report=dict(status='pass',rows=rows,services=services,load_s=load_s,GPU=torch.cuda.get_device_name(),
        source_sha=source_sha,runner_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        text_encoder_file_sha256=hashlib.sha256((args.source/'utils/wan_5b_wrapper.py').read_bytes()).hexdigest(),
        no_video_generated=True,no_online_selector_implemented=True,
        future_prompt_dictionary_allowed_to_online_selector=False,
        limits=['templated development prompts, not general entity matching','labels are constructed prompt roles, not video quality',
            'current-condition summary is a proposed explicit causal-input extension, not existing Q-only interface',
            'no routing hyperparameters promoted from this characterization'])
    (args.output/'condition_retrieval.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',unique_texts=len(texts),queries=len(rows),load_s=load_s)))


if __name__=='__main__':main()
