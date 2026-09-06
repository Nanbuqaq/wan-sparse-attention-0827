#!/usr/bin/env python3
"""Two matched39-latent trajectories: no-op gate plus isolated contract data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.prefetch_context_probe import NextPrototypeObserver
from adapters.longlive_sparse.stats import SparseRunStats
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from scripts.probe_memory_dynamics import prototype_sha


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', choices=('calibration_motion', 'calibration_state'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    protocol_path = ROOT/'configs/system/remaining_contracts_probe.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['latent_frames'] != 39 or protocol['observed_latent_starts'] != [30,33,36]:
        raise ValueError('this bounded implementation requires the frozen39-latent protocol')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU required')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('freeze source before capture')
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = next(p for p in json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates'] if p['prompt_id']==args.prompt)
    frozen = json.loads((ROOT/'configs/formal/system_method_freeze.json').read_text())
    config = next(c for c in frozen['configs'] if c['config_id']=='legacy_final_system')
    system = LongLiveSystemConfig.from_mapping(config['longlive_system'])
    base = yaml.safe_load((ROOT/'configs/inferhub/rag_method_21.yaml').read_text())
    (args.output/'empty.txt').write_text('')
    base.update(data_path=str(args.output/'empty.txt'), output_folder=str(args.output/'base_load'), inference_iter=0)
    base['longlive_system'] = system.as_dict()
    params = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params'][config['method']]
    base['sparse_history'].update(method=config['method'], backend=config['backend'], history_density=.25,
        refresh_policy='per_chunk', rope_policy='upstream_zero', method_params=params, record_per_call=True)
    config_path = args.output/'load.yaml'; config_path.write_text(yaml.safe_dump(base, sort_keys=False))
    os.environ['INFER_OUTPUT_DIR'] = str(args.output.resolve())
    os.environ['LONGLIVE_CAPTURE_COMPLETE_ATTENTION'] = '0'
    os.environ['LONGLIVE_CAPTURE_CASE_TAG'] = args.prompt
    from scripts.run_longlive_sparse import run_config
    from adapters.longlive_sparse.runtime import configure_pipeline_system
    from utils.misc import set_seed
    loaded = time.perf_counter(); pipeline = run_config(config_path)['pipeline']; model_load = time.perf_counter()-loaded
    states, latents = [], []
    for capture in (False, True):
        configure_pipeline_system(pipeline, system)
        pipeline.sparse_history_archive.reset()
        pipeline.sparse_history_aggregate_stats = SparseRunStats(method=config['method'])
        pipeline.sparse_history_completed_runs = []
        observer = NextPrototypeObserver() if capture else None
        for module in pipeline.sparse_history_modules:
            module.clear_selection_cache(); module.clear_capture_state()
            module.memory_dynamics_observer = observer
        os.environ.update(LONGLIVE_CAPTURE_COMPLETE_ATTENTION='1' if capture else '0',
            LONGLIVE_COMPLETE_CAPTURE_LAYERS='0,3', LONGLIVE_COMPLETE_CAPTURE_STARTS='46800,51480,56160',
            LONGLIVE_COMPLETE_CAPTURE_PASSES='1')
        set_seed(20260904)
        noise = torch.randn(1,39,16,60,104,device=next(pipeline.generator.parameters()).device,dtype=torch.bfloat16)
        noise_sha = tensor_sha256(noise)
        started = time.perf_counter()
        _, latent = pipeline.inference(noise=noise, text_prompts=[prompt['prompt']], return_latents=True,
            low_memory=True, profile=False, skip_vae_decode=True)
        torch.cuda.synchronize()
        stats = pipeline.sparse_history_archive.stats.as_dict()
        ordered = [(r['layer_id'], r['current_start'], r['denoising_pass'], r['route_plan_sha256']) for r in stats['call_records']]
        state = {'capture': capture, 'diagnostic_wall_s': time.perf_counter()-started,
            'noise_sha': noise_sha, 'latent_sha': tensor_sha256(latent), 'ordered_routes': ordered,
            'future_archive_prototype_sha': prototype_sha(pipeline.sparse_history_archive),
            'fallback_calls': stats['dense_fallback_calls']}
        name = 'capture' if capture else 'reference'
        torch.save(latent.cpu(), args.output/f'{name}_latents.pt')
        (args.output/f'{name}_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        (args.output/f'{name}_state.json').write_text(json.dumps(state, indent=2)+'\n')
        if capture:
            if len(observer.records) != 12:
                raise ValueError('expected four layers by three consecutive chunks')
            torch.save({'records': observer.records, 'scope': 'causal_source_Q_and_preexisting_target_prototypes'}, args.output/'contexts.pt')
        states.append(state); latents.append(latent.cpu())
    equal = torch.equal(*latents) and all(states[0][k] == states[1][k] for k in ('noise_sha','ordered_routes','future_archive_prototype_sha'))
    captures = list((args.output/'complete_attention_captures'/args.prompt).glob('*.pt'))
    if len(captures) != 6 or any(s['fallback_calls'] for s in states):
        raise ValueError('complete teacher captures missing or fallback occurred')
    result = {'status': 'pass' if equal else 'fail', 'prompt': args.prompt, 'seed': 20260904, 'latent_frames': 39,
        'source_commit': source, 'GPU': torch.cuda.get_device_name(), 'model_load_s': model_load,
        'same_latents_routes_noise_and_future_prototypes': equal, 'context_records': 12, 'complete_teacher_captures': 6,
        'formal_holdouts_used': False, 'new_routes_executed': False, 'speed_claim': False,
        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'missing': 0, 'timeliness_not_estimated_from_instrumented_wall_time': True}
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not equal:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
