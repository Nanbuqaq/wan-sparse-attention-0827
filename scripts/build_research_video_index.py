#!/usr/bin/env python3
"""Integrity-locked index of completed formal and explicitly separate stress videos."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', action='append', nargs=3, metavar=('LABEL', 'VIDEO_ROOT', 'QUALITY_ROOT'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows, sources = [], []
    for label, root_arg, quality_arg in args.batch:
        root, quality_root = Path(root_arg).resolve(), Path(quality_arg).resolve()
        expected = json.loads((root/'control/expected.json').read_text())
        cases = json.loads((root/'recovered_states.json').read_text())['cases']
        wanted = {c['id']: c for c in expected['cases']}
        if {c['id'] for c in cases} != set(wanted) or len(cases) != len(wanted):
            raise ValueError('missing/duplicate terminal video')
        quality = {}
        for qpath in quality_root.glob('lane*/*/quality.json'):
            q = json.loads(qpath.read_text())
            if q.get('metric_input_protocol_version') != 2 or q['status'] != 'pass':
                raise ValueError('invalid quality protocol')
            for entry in q['rows']:
                if entry['case_id'] in quality:
                    raise ValueError('duplicate completed quality result')
                quality[entry['case_id']] = (entry, qpath)
        if set(quality) != set(wanted):
            raise ValueError('not all terminal videos have valid canonical quality')
        stress = bool(expected.get('exclude_from_formal_mean_pareto_selection', False))
        for case in cases:
            if case['status'] != 'pass' or case['case_key'] != wanted[case['id']]['case_key']:
                raise ValueError('expected a passing, unchanged frozen video')
            video = Path(case['video'])
            if sha(video) != case['video_sha256']:
                raise ValueError('video integrity mismatch')
            q, qpath = quality[case['id']]
            poster = Path(q['canonical_render']['render_directory'])/f"detail_{case['pixel_frames']-1:04d}.png"
            if not poster.is_file():
                raise ValueError('canonical endpoint missing')
            artifacts = {'video': video, 'latents': video.parent/'latents.pt',
                'stats': video.parent/'sparse_history_stats.json', 'config': video.parent/'case_config.json',
                'state': video.parent/'case_state.json', 'quality': qpath, 'poster': poster}
            rows.append({'batch': label, 'stress_only': stress, 'case_id': case['id'],
                'config': wanted[case['id']]['formal_config_id'], 'prompt': case['prompt_id'], 'seed': case['seed'],
                'latent_frames': case['latent_frames'], 'pixel_frames': case['pixel_frames'],
                'end_to_end_s': case['end_to_end_s'], 'source_commit': case['case_key']['commit'],
                'lpips': q['lpips_mean'], 'late_lpips': q['late_quarter_lpips_mean'],
                'status': 'pass', 'artifacts': {k: {'path': str(v), 'sha256': sha(v)} for k,v in artifacts.items()}})
        for name in ('control/expected.json','recovered_states.json'):
            path = root/name
            sources.append({'path': str(path), 'sha256': sha(path)})
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'pass', 'videos': len(rows), 'formal_videos': sum(not r['stress_only'] for r in rows),
        'stress_videos': sum(r['stress_only'] for r in rows), 'missing': 0, 'rows': rows, 'sources': sources,
        'MP4_preview_not_model_equivalence': True, 'stress_excluded_from_formal_mean_Pareto_selection': True}
    (args.output/'index.json').write_text(json.dumps(result, indent=2)+'\n')
    def link(path):
        return quote(os.path.relpath(path, args.output), safe='/')
    doc = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>LongLive research video index</title>',
        '<style>body{font-family:system-ui;max-width:1450px;margin:24px auto;background:#f6f7f9;color:#18202b} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}article{background:white;padding:14px;border:1px solid #ddd;border-radius:8px}video{width:100%;height:auto}small{overflow-wrap:anywhere}a{margin-right:10px}</style>',
        f'<h1>LongLive 研究视频与原始证据</h1><p>{result["formal_videos"]}条正式视频与{result["stress_videos"]}条独立stress分开。',
        'MP4仅用于预览；质量分数来自固定VAE重新解码的原始像素。系统等价由全部latent、route和编码前RGB审计证明。',
        '这里不声称绝对质量优于Dense，也不是人工偏好研究。</p>']
    for label, _, _ in args.batch:
        subset = sorted([r for r in rows if r['batch']==label], key=lambda r:(r['prompt'],r['seed'],r['config']))
        doc += [f'<h2>{html.escape(label)}'+(' — stress：不进入正式均值或选择' if subset[0]['stress_only'] else '')+'</h2><div class="grid">']
        for row in subset:
            a = row['artifacts']
            doc += [f'<article><h3>{html.escape(row["prompt"])} · {row["seed"]}</h3>',
                f'<p>{html.escape(row["config"])} · {row["pixel_frames"]} frames · complete {row["end_to_end_s"]:.3f}s</p>',
                f'<video controls preload="none" poster="{link(a["poster"]["path"])}" src="{link(a["video"]["path"])}"></video>',
                f'<p>LPIPS {row["lpips"]:.4f} · late {row["late_lpips"]:.4f}</p><p>']
            doc += [f'<a href="{link(value["path"])}">{name}</a>' for name,value in a.items() if name not in ('poster','latents')]
            doc += [f'</p><small>source {row["source_commit"]}<br>video SHA {a["video"]["sha256"]}</small></article>']
        doc += ['</div>']
    doc += ['</html>']
    (args.output/'index.html').write_text('\n'.join(doc))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources')}))


if __name__ == '__main__':
    main()
