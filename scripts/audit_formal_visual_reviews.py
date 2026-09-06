#!/usr/bin/env python3
"""Bind actual assistant decisions to every unique render and frozen case."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row['prompt'], row['seed'], row['latent_frames']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decisions', type=Path, required=True)
    p.add_argument('--panels', type=Path, required=True)
    p.add_argument('--system-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    decisions = json.loads(args.decisions.read_text())
    panels = json.loads(args.panels.read_text())
    system = json.loads(args.system_audit.read_text())
    if decisions['status'] != 'complete' or not decisions['overview_inspected_for_all_unique_renders']:
        raise ValueError('actual complete visual decisions required')
    if not decisions['not_a_human_preference_study'] or decisions['formal_parameters_changed']:
        raise ValueError('visual review scope or formal freeze was violated')
    if not panels.get('thumbnail_aspect_ratio_preserved') or system['status'] != 'pass':
        raise ValueError('valid panels and system equivalence audit required')
    groups = {key(r): r for r in decisions['groups']}
    panel_groups = {key(r): r for r in panels['groups']}
    system_groups = {key(r): r for r in system['groups']}
    if not (groups.keys() == panel_groups.keys() == system_groups.keys()) or len(groups) != len(decisions['groups']):
        raise ValueError('missing or duplicate visual group')
    rows = []
    for group, decision in sorted(groups.items()):
        if decision['quarters_inspected'] != [1, 2, 3, 4] or len(decision['quarter_observations']) != 4:
            raise ValueError('every quarter needs actual inspection and notes')
        panel = panel_groups[group]
        if sha(panel['panel']) != panel['panel_sha256'] or sha(panel['quality_path']) != panel['quality_sha256']:
            raise ValueError('review evidence changed')
        quality = json.loads(Path(panel['quality_path']).read_text())
        if quality.get('metric_input_protocol_version') != 2:
            raise ValueError('invalid canonical metric protocol')
        if not system_groups[group]['same_final_latents'] or not system_groups[group]['same_final_preencode_pixels']:
            raise ValueError('shared Final review requires full-output equivalence')
        for config in ('rag_dense', 'legacy_final', 'legacy_final_system'):
            grade = decision['rag_dense' if config == 'rag_dense' else 'final_shared']
            for field in ('subject_identity_1to5', 'background_coherence_1to5', 'action_continuity_1to5', 'late_quarter_quality_1to5'):
                if not 1 <= grade[field] <= 5:
                    raise ValueError('invalid visual grade')
            if grade['category_completion_0to2'] not in (0, 1, 2) or grade['state_retention'] not in (
                'stable_sampled', 'sampled_regression', 'uncertain', 'not_applicable'):
                raise ValueError('invalid state/task grade')
            matches = [r for r in quality['rows'] if r.get('formal_config_id') == config]
            if len(matches) != 1:
                raise ValueError('missing canonical config')
            row = matches[0]
            directory = Path(row['canonical_render']['render_directory'])
            sources = [directory/'overview.png']+[directory/f'quarter{i}.png' for i in range(1, 5)]
            sources += [directory/f'detail_{i:04d}.png' for i in decision.get('detail_frames_inspected', [])]
            rows.append({'prompt': group[0], 'seed': group[1], 'latent_frames': group[2],
                'config_id': config, 'case_id': row['case_id'], 'latent_sha256': row['latent_sha256'],
                'grade': grade, 'shared_system_semantic_review': config != 'rag_dense',
                'sources': [{'path': str(s.resolve()), 'sha256': sha(s)} for s in sources],
                'quarter_observations': decision['quarter_observations'], 'events': decision['events'],
                'conclusion': decision['conclusion'], 'face_visibility': decision.get('face_visibility')})
    result = {'status': 'pass', 'reviewed_cases': len(rows), 'unique_trajectories': len({r['latent_sha256'] for r in rows}),
        'missing': 0, 'rows': rows, 'reviewer': decisions['reviewer'], 'not_a_human_preference_study': True,
        'audit_pass_means_complete_review_not_universal_semantic_success': True,
        'new_admission_or_quality_superiority_promoted': False,
        'exclude_from_formal_mean_pareto_selection': bool(decisions.get('exclude_from_formal_mean_pareto_selection', False)),
        'frozen_baselines_and_lossless_system_may_continue_to_independent_length': not decisions.get('exclude_from_formal_mean_pareto_selection', False),
        'decisions_sha256': sha(args.decisions), 'panels_sha256': sha(args.panels), 'system_audit_sha256': sha(args.system_audit)}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
