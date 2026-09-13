#!/usr/bin/env python3
"""Frozen current-request fork on one past text-only bead descriptor.

This measures the existing cue/T5 decision, not video quality or LLM semantics.
Desired compatibility labels are isolated diagnostic annotations, never inputs
to choose_scene. No external service or new model is used.
"""
import argparse,os,sys,json,time,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_scene_admission import SceneDescriptor,choose_scene


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('assets','source','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261010)
    spec=json.loads((ROOT/'configs/system/native_cut_memory_development.json').read_text())
    case=next(x for x in spec['scenarios'] if x['id']=='generated_bead_state_cut_revisit')
    texts={'past_source':case['segments'][1]['prompt'],
        'keep':case['segments'][-1]['prompt'],
        'update':'Back to the same clear cylindrical glass jar on the white table. The jar has now been emptied completely: there are no beads inside or falling into it. Preserve the same jar shape and show its empty transparent interior in a steady medium close-up.',
        'absent':'Back to the same white table. The glass jar has been removed, and no jar or beads remain. Show the empty tabletop in a steady medium close-up.'}
    (args.output/'registration.json').write_text(json.dumps(dict(texts=texts,source_kind='past_prompt_text_only',
        controls=['frozen cue/T5','always abstain','isolated compatibility labels'],
        diagnostic_labels={'keep':True,'update':False,'absent':False},
        online_labels_used=False,threshold_search=False),indent=2)+'\n')
    sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from utils.wan_5b_wrapper import WanTextEncoder
    from adapters.longlive_sparse.strict_checkpoint_init import StrictCheckpointParameterInit
    start=time.perf_counter()
    with StrictCheckpointParameterInit(enabled=True):encoder=WanTextEncoder().to(dtype=torch.bfloat16)
    torch.cuda.synchronize();load_s=time.perf_counter()-start
    prototypes={};timings={}
    for key,text in texts.items():
        torch.cuda.synchronize();start=time.perf_counter()
        encoded=encoder(text_prompts=[text])['prompt_embeds'][0]
        valid=encoded.ne(0).any(-1)
        prototypes[key]=torch.nn.functional.normalize(encoded[valid].float().mean(0),dim=0).cpu()
        torch.cuda.synchronize();timings[key]=time.perf_counter()-start
    candidate=SceneDescriptor(2,48,8.,prototypes['past_source'])
    decisions={key:choose_scene(texts[key],prototypes[key],[candidate],96) for key in ('keep','update','absent')}
    report=dict(status='pass',GPU=torch.cuda.get_device_name(),source_kind='text_only_not_visual_state_witness',
        load_s=load_s,encoding_and_D2H_wall_s=timings,decisions=decisions,
        GPU_peak_bytes=torch.cuda.max_memory_allocated(),CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        prototype_D2H_bytes=sum(t.numel()*t.element_size() for t in prototypes.values()),
        standalone_encoder_cost_not_incremental_shared_T5_cost=True,
        scope='decision diagnostic; same generated source prefix required for subsequent video forks',
        no_local_frozen_LLM_in_registered_runtime=True,not_online_semantic_gate_claim=True)
    torch.save(prototypes,args.output/'prototypes.pt')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
