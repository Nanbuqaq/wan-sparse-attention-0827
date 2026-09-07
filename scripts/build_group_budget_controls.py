#!/usr/bin/env python3
"""Pre-register missing high-budget controls and a new development seed.

Do not regenerate completed seed20260904 Dense/Final25/conditional25 videos.
The 50% controls upper-bound rather than exactly match conditional25 union.
Across free-running trajectories even identical policies can retrieve
different coarse candidates; measure actual bytes, not only nominal ratios.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.case_identity import build_case_identity
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from scripts.build_group_relation_video_suite import build as build_initial


def build(commit, *, latent_frames=120, new_seed=20260908):
    if new_seed == 20260904:
        raise ValueError('new-seed replication must not repeat the original seed')
    old_suites, _ = build_initial(commit, latent_frames)
    suites, expected = {}, []
    for lane, old in old_suites.items():
        templates = {('dense' if c['only_method'] == 'rag_dense' else
                      'final' if c['only_method'] == 'transfer_vaware_hybrid_history' else
                      c['method_params']['relation_admission']): c for c in old['cases']
                     if c['backend'] == 'resident_grouped_fa2'}
        design = [('final', .5, 20260904), ('shared', .5, 20260904),
                  ('dense', 1., new_seed), ('final', .25, new_seed),
                  ('final', .5, new_seed), ('shared', .5, new_seed),
                  ('per_group', .25, new_seed)]
        if lane:
            design.reverse()
        cases = []
        for variant, density, seed in design:
            case = copy.deepcopy(templates[variant])
            case.update(history_density=density, seed=seed)
            method = case['only_method']
            SparseHistoryConfig(method=method, backend=case['backend'], history_density=density,
                                method_params=case['method_params'], refresh_policy='per_chunk')
            system = LongLiveSystemConfig(**case['longlive_system'])
            identity = build_case_identity(commit=commit, method=method, prompt_id=case['prompt_id'],
                prompt=case['prompt'], seed=seed, latent_frames=latent_frames, history_density=density,
                rope_policy='upstream_zero', refresh_policy='per_chunk', backend=case['backend'],
                method_params=case['method_params'], system_identity=system.identity_dict())
            expected.append(dict(identity, lane=lane, variant=variant, density=density, seed=seed))
            cases.append(case)
        suites[lane] = dict(old, cases=cases, methods=list(dict.fromkeys(c['only_method'] for c in cases)),
            status='frozen_group_budget_development_controls',
            comparison='50pct shared upper-budget controls; not exactly matched bytes',
            completed_original_seed_cases_reused=True, new_development_seed=new_seed,
            independent_timing_repeats=False, quality_inference_unit='whole_video',
            promotion='new-seed absolute identity/task review plus actual bytes; no promotion on latent fidelity alone')
    if len(expected) != 14 or len({r['case_key_sha256'] for r in expected}) != 14:
        raise ValueError('invalid budget-control design')
    return suites, expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--latent-frames', type=int, default=120)
    p.add_argument('--new-seed', type=int, default=20260908)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
    suites, expected = build(commit, latent_frames=args.latent_frames, new_seed=args.new_seed)
    if args.validate_only:
        print(json.dumps(dict(status='validated', cases=len(expected))))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shas = {}
    for lane, suite in suites.items():
        raw = json.dumps(suite, indent=2)+'\n'
        (args.output_dir/f'lane{lane}.json').write_text(raw)
        shas[str(lane)] = hashlib.sha256(raw.encode()).hexdigest()
    (args.output_dir/'expected.json').write_text(json.dumps(dict(cases=expected, suite_sha256=shas), indent=2)+'\n')
    print(json.dumps(dict(status='frozen', commit=commit, cases=len(expected), suite_sha256=shas)))


if __name__ == '__main__':
    main()
