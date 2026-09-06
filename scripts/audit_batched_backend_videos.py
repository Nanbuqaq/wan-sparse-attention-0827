#!/usr/bin/env python3
"""Fixed admission/KV budget, changed executor: raw-output and complete-time gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_system_video_comparison import load_case
from adapters.longlive_sparse.offline_eval import output_error_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root).resolve()
    cases = json.loads((root/'recovered_states.json').read_text())['cases']
    rows = []
    for lane in range(4):
        pair = [c for c in cases if c['lane'] == lane]
        if len(pair) != 2 or any(c['status'] != 'pass' for c in pair):
            raise ValueError('eight technical passing videos required')
        old = next(c for c in pair if c['backend'] == 'grouped_fa2')
        new = next(c for c in pair if c['backend'] == 'batched_fa2')
        a, b = load_case(Path(old['video']).parent), load_case(Path(new['video']).parent)
        for key in ('method', 'prompt_sha256', 'seed', 'latent_frames', 'history_density', 'rope_policy', 'refresh_policy', 'method_params'):
            if old['case_key'][key] != new['case_key'][key]:
                raise ValueError(f'admission changed: {key}')
        left_system, right_system = dict(old['case_key']['system']), dict(new['case_key']['system'])
        left_system.pop('execution_dataflow'); right_system.pop('execution_dataflow')
        if left_system != right_system or old['initial_noise_sha256'] != new['initial_noise_sha256']:
            raise ValueError('system budget or initial noise changed')
        exact_latents = torch.equal(a[4], b[4])
        routes = a[5] == b[5]
        raw = old.get('raw_video_rgb_sha256') == new.get('raw_video_rgb_sha256') and old.get('raw_video_rgb_sha256') is not None
        backend_s = [sum(c['backend_complete_s'] for c in item[2]['call_records']) for item in (a, b)]
        result = {'lane': lane, 'method': old['method'], 'prompt': old['prompt_id'],
            'same_noise': True, 'same_latents': exact_latents, 'same_ordered_routes': routes,
            'same_preencode_RGB': raw, 'latent_error': output_error_metrics(a[4], b[4]),
            'same_KV_H2D_bytes': old['transferred_bytes'] == new['transferred_bytes'],
            'original_complete_s': old['end_to_end_s'], 'batched_complete_s': new['end_to_end_s'],
            'complete_reduction': 1-new['end_to_end_s']/old['end_to_end_s'],
            'original_backend_complete_s': backend_s[0], 'batched_backend_complete_s': backend_s[1],
            'complete_backend_reduction': 1-backend_s[1]/backend_s[0],
            'peak_allocated_GiB': [old['peak_allocated_gb'], new['peak_allocated_gb']],
            'case_ids': [old['id'], new['id']]}
        result['equivalence_gate'] = all(result[k] for k in ('same_latents', 'same_ordered_routes', 'same_preencode_RGB', 'same_KV_H2D_bytes'))
        rows.append(result)
    equivalent = all(r['equivalence_gate'] for r in rows)
    performance = all(r['complete_reduction'] >= 0 and r['complete_backend_reduction'] >= .10 for r in rows)
    payload = {'status': 'pass', 'technical_cases': 8, 'missing': 0, 'groups': rows,
        'equivalence_gate': equivalent, 'performance_gate': performance,
        'eligible_for_formal_system_config': equivalent and performance,
        'scope': 'same_card_fixed_admission_backend_specialization',
        'legacy_attention_s_is_not_comparable_to_complete_backend_scope': True,
        'MP4_hash_not_model_equivalence': True,
        'states_sha256': hashlib.sha256((root/'recovered_states.json').read_bytes()).hexdigest()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
