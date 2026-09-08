#!/usr/bin/env python3
"""Exact temporal eligibility geometry for the pinned RAG cache policy.

Potentially retrievable is not the same as selected/attended. This diagnostic
does not claim the intentional budget reservation is an implementation bug.
"""
import argparse
import hashlib
import json
from pathlib import Path
import yaml


def geometry(current,*,chunk,local,sink,memory,recent_exclude):
    if current<local or local<=sink or chunk<1 or recent_exclude<0:raise ValueError('steady-state geometry required')
    before=list(range(sink,current-(local-sink)))
    eligible=before[:max(0,len(before)-recent_exclude)]
    after_local=list(range(current+chunk-(local-sink),current+chunk))
    local_budget=max(0,local-sink-memory)
    exact_local=after_local[-local_budget:] if local_budget else []
    exact=list(range(sink))+exact_local
    guaranteed_excluded=sorted(set(range(current))-set(exact)-set(eligible))
    before_local=set(range(current-(local-sink),current))
    newly_evicted=sorted(before_local-set(after_local))
    return dict(current_latent=current,current_chunk=list(range(current,current+chunk)),
        sink_frames=list(range(sink)),exact_local_including_current=exact_local,
        exact_past=[f for f in exact_local if f<current],
        CPU_pool_before_current_forward=before,coarse_eligible_before_current_forward=eligible,
        newly_evicted_but_not_in_precomputed_coarse_pool=newly_evicted,
        GPU_resident_but_excluded_from_exact=[f for f in after_local if f not in exact_local],
        guaranteed_excluded_past=guaranteed_excluded,
        selected_history_frames_not_inferred=True,
        interpretation='cache capacity, dense attention allocation and coarse eligibility are distinct')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--current-latent',type=int,default=78);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text());m=cfg['model_kwargs']
    result=geometry(args.current_latent,chunk=cfg['num_frame_per_block'],local=m['local_attn_size'],sink=m['sink_size'],
        memory=m['memory_size'],recent_exclude=m.get('recent_exclude',0))
    result.update(status='pass',config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        config=str(args.config.resolve()),source_policy='coarse search before current eviction; local budget = local-sink-memory',
        statistical_quality_or_speed_claim=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
