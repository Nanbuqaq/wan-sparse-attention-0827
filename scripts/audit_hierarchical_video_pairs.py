#!/usr/bin/env python3
"""Same-card full-video equivalence and total KV+index traffic audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_system_video_comparison import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root).resolve()
    cases = json.loads((root/'recovered_states.json').read_text())['cases']
    if len(cases) != 8 or any(c['status'] != 'pass' for c in cases):
        raise ValueError('all eight terminal video cases required')
    groups = []
    for lane in range(4):
        pair = [c for c in cases if c['lane'] == lane]
        old = next(c for c in pair if c['case_key']['system']['gpu_union_cache'] == 'per_chunk')
        new = next(c for c in pair if c['case_key']['system']['gpu_union_cache'] == 'hierarchical')
        if old['initial_noise_sha256'] != new['initial_noise_sha256']:
            raise ValueError('initial noises differ')
        compared = compare(Path(old['video']).parent, Path(new['video']).parent)
        if compared['status'] != 'pass' or not compared['bitwise_equal_latents']:
            raise ValueError('same-route latent byte equivalence failed')
        rows = []
        for case in (old, new):
            stats = json.loads(Path(case['stats']).read_text())
            index_bytes = stats.get('restore_index_h2d_bytes', 0)
            rows.append({'cache_mode': case['case_key']['system']['gpu_union_cache'],
                'complete_time_s': case['end_to_end_s'], 'kv_h2d_bytes': case['transferred_bytes'],
                'index_h2d_bytes': index_bytes, 'kv_plus_index_h2d_bytes': case['transferred_bytes']+index_bytes,
                'cache': stats.get('history_union_cache'), 'timing': stats['timing'],
                'peak_allocated_gb': case['peak_allocated_gb'], 'case_id': case['id']})
        log = root/f'lane{lane}/runner.log'
        hardware = [json.loads(line[8:]) for line in log.read_text().splitlines() if line.startswith('RUNTIME ')]
        if len(hardware) != 1:
            raise ValueError('one loaded same-GPU pair required')
        groups.append({'lane': lane, 'method': old['method'], 'prompt': old['prompt_id'],
            'runtime': hardware[0], 'same_noise_routes_latents': True,
            'same_video_file_bytes': old['video_sha256'] == new['video_sha256'],
            'preencode_RGB_equivalence': 'not recorded in this original batch; diagnostic follow-up required',
            'complete_time_reduction': compared['end_to_end_reduction'],
            'kv_h2d_reduction': 1-rows[1]['kv_h2d_bytes']/rows[0]['kv_h2d_bytes'],
            'kv_plus_index_h2d_reduction': 1-rows[1]['kv_plus_index_h2d_bytes']/rows[0]['kv_plus_index_h2d_bytes'],
            'rows': rows, 'comparison': compared,
            'runner_log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    result = {'status': 'pass', 'technical_cases': 8, 'missing': 0, 'groups': groups,
        'formal_promotion': False, 'outcome': 'mixed_not_promoted',
        'decision': 'retain proven union-only system; physical traffic savings did not yield robust complete-time benefit',
        'no_pooled_cross_prompt_absolute_latency': True,
        'video_pixel_discrepancy_not_attributed_to_KV_or_encoder_without_raw_capture': True,
        'entire_research_plan_complete': False}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'pass', 'groups': [{k: g[k] for k in ('method', 'prompt', 'complete_time_reduction',
        'kv_h2d_reduction', 'kv_plus_index_h2d_reduction')} for g in groups]}, indent=2))


if __name__ == '__main__':
    main()
