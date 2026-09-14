#!/usr/bin/env python3
"""Frozen request robustness on an existing causal descriptor catalog.

Evaluation labels never enter choose_available. This is a decision diagnostic,
not a descriptor parser, new entity resolver or video-quality experiment.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_scene_admission import SceneDescriptor,has_revisit_cue
from adapters.longlive_sparse.payload_aware_scene import choose_available


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('assets','source','catalog-summary','registration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-prefix',action='store_true')
    p.add_argument('--reuse-prototypes',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    registration=json.loads(args.registration.read_text())
    original=json.loads(args.catalog_summary.read_text())
    scene=original['wave2']['scene'];now=registration['at_latent']
    roots={int(k):v for k,v in scene['identity_roots'].items()}
    past={}
    for a in scene['archives']:
        if a['source_end']>now:continue
        segment=next(s for s in reversed(original['segments']) if s['start_latent']<=a['source_end']-8)
        past[a['archive_version']]=dict(text=segment['prompt'],source_end=a['source_end'],source_phase=a['source_phase'])
    available=set(registration['available_versions'])
    if not available<=set(past):raise RuntimeError('invented raw payload version')
    torch.set_num_threads(2);torch.manual_seed(20261010)
    sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from utils.wan_5b_wrapper import WanTextEncoder
    from adapters.longlive_sparse.strict_checkpoint_init import StrictCheckpointParameterInit
    began=time.perf_counter()
    with StrictCheckpointParameterInit(enabled=True):encoder=WanTextEncoder().to(dtype=torch.bfloat16)
    torch.cuda.synchronize();load_s=time.perf_counter()-began
    prototypes={};timings={}
    if args.reuse_prototypes:
        old=json.loads(args.reuse_prototypes.with_name('result.json').read_text())
        if old['catalog_summary_sha256']!=hashlib.sha256(args.catalog_summary.read_bytes()).hexdigest():
            raise RuntimeError('cached past catalog differs')
        prototypes=torch.load(args.reuse_prototypes,weights_only=True,map_location='cpu')
    texts={f'past_{v}':x['text'] for v,x in past.items()}
    key_prefix='native_prefix_' if args.native_prefix else ''
    texts.update({key_prefix+x['id']:('The scene transitions. ' if args.native_prefix else '')+x['text'] for x in registration['requests']})
    texts['calibration_original_current']=original['prompts_per_block'][now//8]
    for key,text in texts.items():
        if key in prototypes:continue
        torch.cuda.synchronize();began=time.perf_counter()
        encoded=encoder(text_prompts=[text])['prompt_embeds'][0];valid=encoded.ne(0).any(-1)
        prototypes[key]=torch.nn.functional.normalize(encoded[valid].float().mean(0),dim=0).cpu()
        torch.cuda.synchronize();timings[key]=time.perf_counter()-began
    raw=[SceneDescriptor(v,x['source_end'],x['source_phase'],prototypes[f'past_{v}']) for v,x in past.items()]
    canonical=[SceneDescriptor(v,x['source_end'],x['source_phase'],prototypes[f'past_{roots[v]}']) for v,x in past.items()]
    rows=[]
    for request in registration['requests']:
        decisions={}
        for name,catalog in [('raw_max',raw),('canonical_max',canonical)]:
            decision=choose_available(texts[key_prefix+request['id']],prototypes[key_prefix+request['id']],catalog,available,now,
                ranking='max_similarity',margin=.05,minimum_cosine=.8,min_gap=32)
            selected=decision['selected_version']
            decision['selected_root']=None if selected is None else roots[selected]
            decisions[name]=decision
        rows.append(dict(id=request['id'],triggered=has_revisit_cue(request['text']),decisions=decisions,
                         evaluation_only=request['evaluation_only']))
    torch.save(prototypes,args.output/'prototypes.pt')
    calibration=choose_available(texts['calibration_original_current'],prototypes['calibration_original_current'],canonical,available,now,
        ranking='max_similarity',margin=.05,minimum_cosine=.8,min_gap=32)
    expected=next(x for x in scene['decisions'] if x['at_latent']==now and x.get('scores'))
    expected_scores={x['archive_version']:x['cosine'] for x in expected['scores']}
    score_error=max(abs(x['cosine']-expected_scores[x['archive_version']]) for x in calibration['scores'])
    result=dict(status='pass',rows=rows,current_frame=now,available_versions=sorted(available),
        native_transition_prefix_in_current_encoding=args.native_prefix,
        original_logged_decision_calibration=calibration,original_logged_score_max_abs_error=score_error,
        original_decision_reproduced=calibration['selected_version']==expected['selected_version'],
        catalog_summary_sha256=hashlib.sha256(args.catalog_summary.read_bytes()).hexdigest(),
        registration_sha256=hashlib.sha256(args.registration.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        GPU=torch.cuda.get_device_name(),load_s=load_s,encoding_and_D2H_wall_s=timings,
        retained_CPU_prototype_bytes=sum(t.numel()*t.element_size() for t in prototypes.values()),
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        online_inputs='current text and its T5 summary; past prompt descriptors and recorded causal roots; payload availability',
        evaluation_labels_used_online=False,threshold_or_prompt_search=False,new_videos=0,
        S1_parser_available=False,S1_quality_claim=False,
        limitations=['same real past catalog, independent current texts; not new causal generation sequences',
            'past prompt is requested content, not an observed-state certificate',
            'standalone T5 loading/encoding cost is not incremental cost when conditions are already encoded',
            'no suitable frozen local parser in registered runtime; no per-object string patch introduced'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='pass',rows=rows)),flush=True)


if __name__=='__main__':main()
