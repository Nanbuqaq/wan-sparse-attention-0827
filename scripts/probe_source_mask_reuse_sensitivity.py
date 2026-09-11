#!/usr/bin/env python3
"""CPU route sensitivity to causal mask reuse; no quality or runtime claims."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_native_source_foreground import source_token_masks
from adapters.longlive_sparse.native_oracle_source_mask import fixed_mask_indices
from adapters.longlive_sparse.stable_source_mask_fill import stable_mask_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--source-start', type=int, required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--input-kind', choices=('raw_stream', 'lossy_decoded'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    masks = np.load(args.masks)['pixel_masks'].astype(np.bool_)
    full = source_token_masks(masks, args.source_start)
    routes = {}
    for name in ('uniform_midpoint', 'fixed_bit_reversal'):
        def select(mask):
            ids = np.flatnonzero(mask.flatten())
            if name == 'uniform_midpoint':
                return fixed_mask_indices(ids, source_tokens=mask.size, budget=args.budget, mode='foreground')
            return stable_mask_indices(ids, source_tokens=mask.size, budget=args.budget)
        baseline = set(select(full).tolist())
        rows = []
        for stride in (1, 4, 8, 32):
            sample_indices = np.arange(len(masks)) // stride * stride
            reused = masks[sample_indices]
            tokens = source_token_masks(reused, args.source_start)
            current = set(select(tokens).tolist())
            rows.append(dict(stride=stride, segmented_frames=len(np.unique(sample_indices)),
                mask_tokens=int(tokens.sum()),
                pixel_IoU=float((reused & masks).sum() / (reused | masks).sum()),
                token_IoU=float((tokens & full).sum() / (tokens | full).sum()),
                route_IoU=len(current & baseline) / len(current | baseline),
                route_removed=len(baseline - current)))
        routes[name] = rows
    report = dict(input_masks=str(args.masks), input_kind=args.input_kind,
        source_start=args.source_start, budget=args.budget, routes=routes,
        future_masks_never_used=True, no_video_quality_or_runtime_claim=True,
        note='fixed priority changes the baseline too; use a factorial video comparison')
    (args.output / 'screen.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(routes))


if __name__ == '__main__':
    main()
