#!/usr/bin/env python3
"""Real GPU allocation gate using synthetic native-shape preencoded text values."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.native_shared_conditioning import SharedNativeConditioning
from scripts.run_longlive2_native_reference import CachedNativeTextEncoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU required')
    spec = importlib.util.spec_from_file_location('native_prompt_conditioning',
        ROOT/'third_party/LongLive2/utils/prompt_conditioning.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    gen = torch.Generator().manual_seed(20260911)
    values = {str(i): torch.randn(1, 512, 4096, dtype=torch.bfloat16, generator=gen) for i in range(7)}
    encoder = CachedNativeTextEncoder(values, 'cuda')
    rows = []
    for count in (16, 91, 451):
        prompts = [[str(i % 7) for i in range(count)]]
        for mode in ('reference', 'shared'):
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            before = torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            if mode == 'reference':
                aggregate, blocks = native.encode_prompt_blocks(encoder, prompts, 1)
            else:
                shared = SharedNativeConditioning(encoder)
                aggregate, blocks = shared.encode(encoder, prompts, 1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            retained = torch.cuda.memory_allocated()-before
            peak = torch.cuda.max_memory_allocated()-before
            unique_storage = {x['prompt_embeds'].untyped_storage().data_ptr():
                x['prompt_embeds'].untyped_storage().nbytes() for x in blocks}
            for p, block in zip(prompts[0], blocks):
                if not torch.equal(block['prompt_embeds'].cpu(), values[p]):
                    raise RuntimeError('actual GPU block differs from preencoded reference')
            rows.append(dict(blocks=count,mode=mode,actual_all_block_values_exact=True,
                retained_allocator_bytes=retained,peak_allocator_delta_bytes=peak,
                unique_owned_storage_bytes=sum(unique_storage.values()),
                single_synchronized_prepare_s=elapsed,
                ledger=shared.audit() if mode=='shared' else None))
            del aggregate,blocks,block
            if mode == 'shared':
                del shared
    report = dict(status='pass',GPU=torch.cuda.get_device_name(),torch=torch.__version__,rows=rows,
        native_shape=[1,512,4096],unique_prompts=7,
        input_kind='synthetic BF16 values; real CUDA allocations and native reference preparation',
        full_video_equivalence_not_checked_by_this_component=True,
        timing_is_single_component_observation_not_speedup=True)
    (args.output/'allocation_gate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
