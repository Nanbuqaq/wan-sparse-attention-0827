# Sprint progress (UTC)

## 14:21–15:00: exact control reuse and a real producer pilot

Goal remains faster-and-better **streaming** generation through information grouping, selective use and lifetime-aware organization; systems engineering must support that, not become the only goal.

### Route-control reuse

- Runtime commit: `c669da2540637ca1430cb080cfc2e8caf81bba7e`, pushed to the existing research branch.
- Exact CPU snapshots validate all six route tensor values before reusing digests. This detects inference-tensor and NumPy-alias mutations, unlike identity/version-only caching. Cache epochs/storage/positions/device/layout keys remain intact.
- Only the current per-layer identity snapshot is retained; additional measured CPU storage ~5.84 MB for Dense / 1.80 MB for Final per tested layer.
- CPU-only complete five-call identity window speedups: Dense motion/state 5.65×/5.91×; Final 4.80×/4.76×. Includes initial construction, four hits, and two subsequent digest consumers. Not video speedup.
- Real CUDA self-attention five-call gate: Dense 171.53→117.54 ms; Final 106.11→88.99 ms, exact output and route. Preliminary grouped-policy ordering, not independent randomized timing repetitions.
- Regression at this point: 354 passed, 1 skipped; new suite tests subsequently 12 passed. Initial pytest plugin attempted a forbidden socket before tests, so plugin autoload was disabled; no shared environment was modified.

### Complete development videos

Frozen execution worktree `/tmp/longlive-metadata-sprint.inAL2b/checkout`.
8/8 153-frame videos complete; all four pairs have identical ordered routes, full latents, raw RGB, initial noise and H2D bytes.

| Method / development category | Baseline full s | Reuse full s | Speedup |
|---|---:|---:|---:|
| Dense / motion | 66.712 | 57.386 | 1.163× |
| Dense / state | 64.451 | 52.872 | 1.219× |
| Final / motion | 59.840 | 53.970 | 1.109× |
| Final / state | 54.835 | 58.193 | 0.942× |

Final/state remains a negative single-pair result, not discarded as noise. Route/control and backend component times decreased in that pair, but the full workflow did not. Opposite orders across prompts do not replace same-prompt independent timing repeats. Longer development/cross-hardware checks are diagnostic, not formal promotion.

Facts: `../../results/metrics/sprint24h_20260907/metadata153_audit_v1.json`, adjacent `metadata153_frozen_v1`, and `../../results/videos/sprint24h_20260907/metadata153_v1`.

### Bounded persistent CPU producer

New `page_pipeline.py` directly packs separate frame-major CPU K/V into bounded pinned slots, launches H2D on its own stream, and uses recorded consumption events to return finite GPU slots. No prepacked whole-candidate staging or full-history GPU shadow. Raw-history RoPE and routing are not included in this pilot.

- Three modes: serial, same-thread asynchronous, independent persistent producer. All tested outputs bitwise equal; BF16 versus FP32 relative L2 ~0.0022.
- Q4680, 6 history frames, Page256/cap4/high reuse: serial34.96 ms, same-thread34.74 ms, producer34.57 ms. No meaningful demonstrated gain.
- Q1560, Page64/cap2/medium reuse: serial85.86 ms, same-thread85.75 ms, producer93.42 ms. Negative.
- Initial Q4680 trace: 42 H2D/66.06 MB including padding, H2D2.965 ms, actual copy/kernel overlap0.522 ms, parent46.28 ms / GPUbusy14.09 ms. This trace accidentally retained Nsys default device-event tracing; preserve it, but recheck representative trace with `--cuda-event-trace=false` and use unprofiled timing before conclusions.
- `complete_backend=0` in the older auditor only means it does not recognize this new marker; not zero backend work.
- Next hypothesis: page-level consumer dispatch/softmax merge overhead dominates; evaluate batched/captured consumption before adding more producer threads. Do not mistake overlapping a small transfer for reducing the critical path.
