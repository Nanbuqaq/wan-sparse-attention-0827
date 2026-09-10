#!/usr/bin/env python3
"""CPU-only payload/route audit and own-source video boards for explicit cases."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    import av
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from adapters.longlive_sparse.history_cache import tensor_sha256
    from scripts.build_video_review_storyboards import storyboard
    from scripts.review_native_memory_study import native_review_indices

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    ids = [case['id'] for case in spec['cases']]
    if len(set(ids)) != len(ids) or any(not x.replace('_', '').replace('-', '').isalnum() for x in ids):
        raise ValueError('unique simple case IDs required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows, panels = [], []
    for case in spec['cases']:
        case_root = Path(case['path'])
        row = dict(id=case['id'], path=str(case_root), semantic_status='pending_own_source_visual_review')
        try:
            summary_path = case_root / 'summary.json'
            data = json.loads(summary_path.read_text())
            row.update(summary_sha256=digest(summary_path), generation_status=data['status'])
            if data['status'] != 'pass':
                raise ValueError('generation failed: ' + str(data.get('traceback', data.get('error'))))
            if data['latent_shape'] != [1, 128, 48, 44, 80] or data['fallback_allowed']:
                raise ValueError('qualified original-resolution native protocol required')
            latent = torch.load(case_root / 'latents.pt', weights_only=True, map_location='cpu')
            if list(latent.shape) != data['latent_shape'] or tensor_sha256(latent) != data['latent_sha256']:
                raise ValueError('actual latent payload/summary mismatch')
            for key in ('prefix_reference', 'full_reference'):
                if key not in case:
                    continue
                reference_root = Path(case[key])
                reference = torch.load(reference_root / 'latents.pt', weights_only=True, map_location='cpu')
                rd = json.loads((reference_root / 'summary.json').read_text())
                for field in ('noise_sha256', 'seed', 'cut_scenario', 'assets_manifest_sha256', 'fixed_native_adaln_recipe'):
                    if data[field] != rd[field]:
                        raise ValueError('reference identity differs: ' + field)
                limit = 96 if key == 'prefix_reference' else 128
                equal = torch.equal(latent[:, :limit], reference[:, :limit])
                row[key + '_actual_latent_exact'] = equal
                if not equal:
                    raise ValueError('actual reference latent mismatch: ' + key)
                if key == 'full_reference' and data['pixels']['raw_RGB_sha256'] != rd['pixels']['raw_RGB_sha256']:
                    raise ValueError('full raw RGB reference mismatch')
                del reference
            del latent
            frames, boards = [], {}
            decoded_digest = hashlib.sha256()
            prefix_digest = hashlib.sha256()
            with av.open(str(case_root / 'video.mp4')) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                fps = float(stream.average_rate)
                for index, frame in enumerate(container.decode(video=0)):
                    pixels = frame.to_ndarray(format='rgb24').tobytes()
                    decoded_digest.update(pixels)
                    if index < 381:
                        prefix_digest.update(pixels)
                    frames.append(frame.reformat(width=208, height=120).to_ndarray(format='rgb24'))
                    if index in native_review_indices()[0]:
                        path = args.output / f'{case["id"]}_native{index}.png'
                        frame.to_image().save(path)
                        boards[f'native{index}'] = path.name
            if len(frames) != 509 or data['pixels']['frames'] != 509 or fps != 24:
                raise ValueError('decoded video length/fps differs from protocol')
            if 'prefix_reference' in case:
                reference_digest = hashlib.sha256()
                with av.open(str(Path(case['prefix_reference'])/'video.mp4')) as container:
                    container.streams.video[0].codec_context.thread_count = 2
                    count = 0
                    for index, frame in enumerate(container.decode(video=0)):
                        if index >= 381:
                            break
                        reference_digest.update(frame.to_ndarray(format='rgb24').tobytes())
                        count += 1
                if count != 381 or reference_digest.hexdigest() != prefix_digest.hexdigest():
                    raise ValueError('actual decoded RGB prefix differs from reference')
                row['prefix_reference_decoded_RGB_exact'] = True
            periods = dict(source=(157, 189), away=(253, 381), first_return=(381, 413), late=(413, 509))
            for name, (start, end) in periods.items():
                path = args.output / f'{case["id"]}_{name}.png'
                storyboard(frames, np.linspace(start, end-1, 16).round().astype(int), path)
                boards[name] = path.name
            for page in range(4):
                path = args.output / f'{case["id"]}_away_all{page}.png'
                storyboard(frames, np.arange(253+32*page, 285+32*page), path)
                boards[f'away_all{page}'] = path.name
            panel = Image.new('RGB', (8*224, 158), 'white')
            draw = ImageDraw.Draw(panel)
            draw.text((4, 2), case['id']+' | SOURCE / AWAY / FIRST RETURN / LATE', fill='black')
            for column, index in enumerate(native_review_indices()[1]):
                panel.paste(Image.fromarray(frames[index]).resize((224, 123)), (column*224, 30))
                draw.text((column*224+3, 17), str(index), fill='black')
            panels.append(panel)
            del frames
            row.update(status='technical_pass', decoded_frames=509, fps=fps,
                decoded_RGB_sha256=decoded_digest.hexdigest(), boards=boards,
                payload_sha256={name: digest(case_root/name) for name in ('video.mp4', 'latents.pt')})
            for key in ('runner_commit', 'gpu', 'seed', 'cut_scenario', 'latent_sha256', 'native_DiT_s',
                        'native_VAE_s', 'load_s', 'wall_including_loading_s', 'generation_peak_allocated_bytes',
                        'native_positive_and_negative_KV_bytes', 'native_inplace_cache', 'causal_block_config'):
                row[key] = data.get(key)
            memory = data.get('causal_block_memory')
            resident = data.get('resident_history')
            if resident:
                dispatches = resident['rows']
                row['resident_config'] = resident['config']
                row['executed_attention_density'] = sum(r['logical_pairs'] for r in dispatches)/sum(r['full_native_pairs'] for r in dispatches)
                row['resident_summary_GPU_bytes'] = resident['summary_GPU_peak_bytes']
                row['resident_summary_build_host_s'] = resident['summary_build_host_s']
                row['resident_dispatches'] = len(dispatches)
                row['resident_reused_routes'] = sum(r['route_reused'] for r in dispatches)
                row['resident_ledger'] = {name:sum(r[name] for r in dispatches) for name in
                    ('selection_host_s','prepare_host_s','history_KV_H2D_bytes','index_H2D_bytes','index_select_KV_write_bytes')}
                row['resident_scope'] = 'no CPU raw archive/onload; same physical native KV allocation; summaries are extra GPU storage'
                row['cost_limits'] = ['host scopes include readiness and are not pure CPU arithmetic',
                    'index-select write bytes are logical payload, not HBM counters',
                    'recent control shares the v1 summary machinery and is not an optimized metadata-only recent implementation']
            if memory:
                dispatches = memory['rows']
                row['executed_attention_density'] = sum(r['logical_pairs'] for r in dispatches)/sum(r['native_pairs'] for r in dispatches)
                row['active_source_frames'] = sorted({r['frame'] for r in dispatches if r['active_source']})
                row['block_ledger'] = memory['ledger']
                row['scene_ledger'] = memory['scene_selection']['ledger']
                route_path = case_root/'causal_block_routes.pt'
                if digest(route_path) != data['causal_block_routes']['sha256']:
                    raise ValueError('route payload SHA mismatch')
                routes = torch.load(route_path, weights_only=True, map_location='cpu')['records']
                phases=[0,2] if data['causal_block_config'].get('refresh')=='phase2' else [0]
                expected_routes={(layer,phase) for layer in range(30) for phase in phases}
                if (row['active_source_frames'] != [96] or len(routes) != len(expected_routes)
                    or {(r['layer'],r.get('phase',0)) for r in routes} != expected_routes):
                    raise ValueError('unexpected installation time/layer coverage')
                selected = int(7040 * data['causal_block_config']['fraction'])
                for route in routes:
                    indices = route['source_indices']
                    if indices is None:
                        if selected != 7040:
                            raise ValueError('partial route lacks source indices')
                    elif (list(indices.shape) != [24, selected] or int(indices.min()) < 0 or int(indices.max()) >= 7040
                          or not torch.all(indices[:, 1:] > indices[:, :-1])):
                        raise ValueError('source indices invalid, duplicated, or noncanonical')
                row['route_index_payload_valid'] = True
                if memory.get('oracle'):
                    from adapters.longlive_sparse.native_oracle_source_mask import fixed_mask_indices
                    mask_path = Path(case['oracle_mask_path'])
                    if digest(mask_path) != memory['oracle_mask_sha256']:
                        raise ValueError('oracle mask SHA differs from executed mask')
                    mask = torch.load(mask_path, weights_only=True, map_location='cpu')
                    expected = fixed_mask_indices(mask['indices'], source_tokens=7040,
                        budget=selected, mode=memory['oracle_mask_mode'])
                    if (memory['automatic_online_method'] or not memory['actual_source_latents_verified']
                        or memory['oracle_used_layers'] != list(range(30))):
                        raise ValueError('oracle source provenance/layer coverage not verified')
                    for route in routes:
                        if not torch.equal(route['source_indices'], expected[None].expand(24, -1)):
                            raise ValueError('executed oracle route differs from source-only selection')
                    producer_path = Path(case['oracle_producer_report'])
                    producer = json.loads(producer_path.read_text())
                    if (producer['source_latent_sha256'] != mask['source_latent_sha256']
                        or not producer['source_pixels_only'] or producer['return_or_future_pixels_supplied_to_predictor']):
                        raise ValueError('oracle producer/source identity mismatch')
                    row['oracle'] = {k: memory[k] for k in (
                        'method_variant', 'oracle', 'automatic_online_method', 'oracle_mask_sha256',
                        'oracle_mask_mode', 'actual_source_latents_verified', 'oracle_used_layers',
                        'foreground_source_tokens', 'selected_source_tokens', 'mask_CPU_tensor_bytes',
                        'segmentation_generation_cost_external_not_in_this_model_timing')}
                    row['oracle'].update(executed_indices_equal_to_mask_rule=True,
                        source_latent_sha256=mask['source_latent_sha256'], online_Pareto_eligible=False,
                        external_preprocessing={k:producer[k] for k in (
                            'manual_bbox','CPU_decode_s','model_load_s','all_source_mask_wall_s',
                            'peak_GPU_allocated_bytes','GPU','complete_online_VAE_segmentation_routing_cost_not_measured')},
                        producer_report_sha256=digest(producer_path))
                partition_info=data['causal_block_routes'].get('partition_snapshot')
                if partition_info:
                    path=case_root/'source_key_partitions.pt'
                    if digest(path)!=partition_info['sha256']:raise ValueError('partition snapshot SHA differs')
                    parts=torch.load(path,weights_only=True,map_location='cpu')['records'];lookup={}
                    for part in parts:
                        indices,counts=part['indices'],part['counts']
                        if (list(indices.shape)!=[128,55] or counts.tolist()!=[55]*128
                            or not torch.equal(indices.flatten().sort().values,torch.arange(7040,dtype=indices.dtype))):
                            raise ValueError('archive partition coverage/count mismatch')
                        lookup[(part['archive_version'],part['layer'])]=indices
                    for route in routes:
                        groups=[set(g) for g in lookup[(route['archive_version'],route['layer'])].tolist()]
                        for indices in route['source_indices'].tolist():
                            kept=set(indices);overlap=[len(kept&g) for g in groups]
                            if sum(overlap)!=selected or sum(0<x<55 for x in overlap)>1:
                                raise ValueError('selected raw coordinates do not follow the saved groups')
                    row['retained_archive_partitions_valid']=len(parts)
                    row['selected_group_membership_verified']=True
                row['budget_denominator'] = 'one selected 8-frame 7040-token source per head; full archive still retained'
                row['cost_limits'] = ['host scope times include readiness',
                    'GPU total allocator peak includes temporaries but no stage-local breakdown',
                    'no measured all-transfer trace']
                if memory.get('memory_samples'):
                    row['memory_samples'] = memory['memory_samples']
                    row['cost_limits'].append('RSS is process lifetime peak including loading, not isolated archive RSS')
                else:
                    row['cost_limits'].append('no CPU RSS measurement')
                if 'score_result_D2H_bytes' not in memory['ledger']:
                    row['cost_limits'].append('v1 score-result D2H bytes not included in ledger')
        except Exception as error:
            row.update(status='fail', error=repr(error))
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('id', 'status')}, ensure_ascii=False), flush=True)
    if panels:
        combined = Image.new('RGB', (panels[0].width, sum(p.height for p in panels)), 'white')
        offset = 0
        for panel in panels:
            combined.paste(panel, (0, offset)); offset += panel.height
        combined.save(args.output/'comparison.jpg', quality=95)
    report = dict(schema='native_research_review_v1', rows=rows, manifest_sha256=digest(args.manifest),
        script_sha256=digest(__file__), semantic_review_complete=False, blind_human_review=False,
        status='technical_pass' if rows and all(r['status']=='technical_pass' for r in rows) else 'fail')
    (args.output/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    sections = ['<h1>Native research wave: own-source review</h1><img width="100%" src="comparison.jpg">']
    for row in rows:
        sections.append('<p>'+html.escape(row['id']+': '+row['status'])+' '+
            ' | '.join(f'<a href="{html.escape(path)}">{html.escape(name)}</a>' for name,path in row.get('boards',{}).items())+'</p>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Native wave review</title>'+''.join(sections))
    if report['status'] != 'technical_pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
