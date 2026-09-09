#!/usr/bin/env python3
"""Source-derived frame lineage checked against the completed hybrid pin records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_native_return_context import simulate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for seed in (20260925,20260926):
        path=args.root/f'seed{seed}'/'summary.json';d=json.loads(path.read_text())
        assert d['status']=='pass' and d['latent_shape']==[1,128,48,44,80] and d['local_frames']==32
        assert d['expected_scene_cut_block_indices']==[2,6,12]
        m=d['causal_scene_memory'];assert len(m['installations'])==1
        plan=m['installations'][0]['installation']['admission_plan']
        assert plan['source_frames']==list(range(40,48)) and plan['target_start']==96
        shadow=simulate('shot',cut_starts=(16,48,96))
        assert shadow['native_shot_pin_events']==d['native_shot_pin_events']
        assert all(r['start']==7040 and r['length']==7040 for r in d['final_native_pinned_slots'])
        calls=[]
        for r in shadow['calls'][12:]:
            calls.append(dict(**r,direct_original_source_latents=len(set(r['visible_frame_ids']).intersection(range(40,48)))))
        rows.append(dict(seed=seed,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),calls=calls))
    report=dict(status='pass',rows=rows,source_derived_not_actual_KV_tensor_lineage_capture=True,
        actual_pin_metadata_matches=True,virtual_key_positions_do_not_change_real_source_frame_ids=True,
        source_information_may_persist_in_generated_return_KV=True,
        source_eviction_is_not_proof_of_late_failure_cause=True,not_attention_probabilities=True)
    with (args.output/'lineage.json').open('x') as h:json.dump(report,h,indent=2);h.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
