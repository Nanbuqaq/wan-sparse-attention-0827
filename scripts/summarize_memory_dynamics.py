#!/usr/bin/env python3
"""Audited finite-window two-level memory dynamics, with explicit evidence limits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.memory_dynamics import compare_coordinates
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'max': None}
    return {'n': len(values), 'min': values[0], 'median': statistics.median(values), 'max': values[-1]}


def summarize(root):
    root = Path(root)
    terminal = json.loads((root/'terminal.json').read_text())
    if terminal['status'] != 'pass':
        raise ValueError('complete passing diagnostic required')
    observations = json.loads((root/'route_observations/observations.json').read_text())['records']
    phases = defaultdict(list)
    for row in observations:
        if row['fresh_shadow_computed']:
            phases[(row['layer'], row['phase'])].append(row)
    denoising = [{'layer': layer, 'phase': phase,
                 'token_jaccard': distribution([r['executed_vs_fresh_tokens']['jaccard'] for r in rows]),
                 'block_jaccard': distribution([r['executed_vs_fresh_blocks']['jaccard'] for r in rows]),
                 'fresh_mass_not_measured': True}
                for (layer, phase), rows in sorted(phases.items())]
    by_chunk, by_layer = defaultdict(dict), defaultdict(list)
    for path in sorted((root/'route_observations').glob('layer*_pass00.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        row = payload['record']
        plan = HistoryRoutePlan.from_state_dict(payload['executed_route'])
        by_chunk[row['current_start']][row['layer']] = plan
        by_layer[row['layer']].append((row['current_start'], plan))
    layers = []
    for start, plans in sorted(by_chunk.items()):
        edges = []
        for left in sorted(plans):
            if left + 1 in plans:
                metric = compare_coordinates(plans[left], plans[left+1], token_base=1560)
                edges.append({'left':left,'right':left+1, **metric})
        layers.append({'current_start': start, 'actual_adjacent_edges':edges,
                       'four_layer_0_3_median':statistics.median([r['jaccard'] for r in edges if r['left']<3])
                            if all(x in plans for x in range(4)) else None})
    fine = []
    for layer, rows in sorted(by_layer.items()):
        previous = None
        accesses = defaultdict(list)
        counts = Counter()
        adjacent = []
        for start, plan in sorted(rows):
            if previous is not None:
                adjacent.append(compare_coordinates(previous,plan,token_base=1560,block_size=64)['jaccard'])
            previous = plan
            f, t = plan.union_frame_ids, plan.union_token_ids
            for head in range(f.shape[1]):
                valid = f[0,head] >= 0
                blocks = torch.unique(f[0,head,valid]*25 + t[0,head,valid]//64)
                for block in blocks.tolist():
                    accesses[(head,block)].append(start//4680)
                    counts[(head,block)] += 1
        gap = [b-a for times in accesses.values() for a,b in zip(times,times[1:])]
        ordered = sorted(counts.values(), reverse=True)
        top = max(1, (len(ordered)+9)//10)
        fine.append({'layer':layer,'selected_blocks_seen':len(counts),
                     'adjacent_block_jaccard':distribution(adjacent),
                     'revisit_gap_chunks':distribution(gap),
                     'top10pct_blocks_access_share':sum(ordered[:top])/sum(ordered) if ordered else None,
                     'right_censored':True,'lru_saved_time_not_measured':True})
    calls = json.loads((root/'generator_calls.json').read_text())
    archive = [{'committed_latents': row['current_start']//1560+3, **row['archive_storage']}
               for row in calls if 'archive_storage' in row]
    nonempty = [r for r in archive if r['kv_bytes']>0]
    slope = None
    if len(nonempty)>1:
        a,b = nonempty[0],nonempty[-1]
        slope = (b['kv_bytes']-a['kv_bytes'])/(b['committed_latents']-a['committed_latents'])
    return {'status':'pass','input':str(root.resolve()),'source_commit':terminal['source_commit'],
            'prompt_id':terminal['prompt']['prompt_id'],'seed':terminal['seed'],
            'latent_frames':terminal['latent_frames'],'denoising_axis':denoising,
            'layer_axis':layers,'fine_chunk_lifetime':fine,
            'coarse_lifecycle':json.loads((root/'coarse_lifecycle.json').read_text()),
            'archive_growth':{'rows':archive,'observed_bytes_per_additional_latent':slope,
                              'bounded_total_archive':False},
            'pulses':terminal['pulses'],
            'limitations':['shadow route similarity does not prove output equivalence',
                           'three-chunk latent divergence is not a video quality ranking',
                           'finite-window revisit counts are censored and not eviction guarantees',
                           'diagnostic service times are not end-to-end performance estimates'],
            'terminal_sha256':hashlib.sha256((root/'terminal.json').read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=summarize(args.input)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')
    print(json.dumps({'status':'pass','prompt':result['prompt_id'],
                      'archive_bytes_per_latent':result['archive_growth']['observed_bytes_per_additional_latent'],
                      'denoising_axis':result['denoising_axis']},indent=2))


if __name__=='__main__':
    main()
