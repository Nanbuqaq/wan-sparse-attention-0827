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

## 15:00–16:00: conditional groups and organized execution

### Grouping evidence, before video promotion

Eight complete development captures: two categories × layers0/9/19/29, all at latent start30/pass0. Thirteen routes per capture: legacy Final; random-balanced/spatial-quadrant/Q-feature grouping × shared/per-group admission ×25%/50% history pair budgets. The online selector only consumes Q summaries, CPU prototypes and coordinates; all routes are built before the full exact/current/recent teacher is computed.

- Random grouping barely changes per-group selection. At layer0, spatial grouping also gives little benefit.
- Spatial per-group25 versus the same spatial shared25 proxy lowers relative-L2 error at layers9/19/29 in both categories. Transfer unions expand to ~34–40%; these are equal-pair, **not equal-byte**, comparisons.
- At layer29, motion spatial shared/per-group relativeL2 .06670→.05435, Q-feature .06310→.04921; state .06192→.05741 and .06024→.05287. Q-feature unions ~45–48%, larger than spatial.
- Q-feature grouping is not consistently superior: at layer9 spatial gives better error/transfer trade-offs; layer0 motion Q-feature conditional is worse. Do not cherry-pick only layer29.
- New opt-in development method `group_relation_history`: legacy Final for layers0–7, declared grouping/admission for layers8–29. Both early fallback and active route branches passed real-CUDA full-context FP32 gates. No video quality conclusion yet.

### Execution and pipeline evidence

- Resident metadata + direct strided exact/history packing avoids intermediate per-group concatenations, but retains final varlen KV replication. Nine GPU layout/batch/zero-exact/padding gates passed bitwise packing and FA2 output equality. Dynamic token counts and strides are runtime kernel arguments to avoid per-route recompilation.
- Actual group-method Q4680 five-forward gate: grouped196.00 ms → resident148.27 ms, same route/output; peak allocation1.281→1.086 GB. Dense and legacy Final receive the same backend opportunity and passed their branch gates. These remain synthetic full-forward timings, not video results.
- CUDA Graph page consumption (static replay addresses): Q4680/Page256 serial16.82 ms → asynchronous13.63 ms; dedicated producer13.65 ms adds no independent improvement. Page64/Q1560 producer remains negative. Graph build ~1.5 s and extra graph allocation~29.2 MB are reported, not hidden.
- Important correction: initial graph-level Nsight omitted graph kernels. Old `page_graph_q4680_v1.activity.json` is invalid for GPU busy/idle/complete overlap; a neighboring erratum preserves this. Auditors now reject aggregate graph traces.
- Correct node-level representative trace: parent13.884 ms; GPUbusy13.229 ms; Attention6.145 ms; H2D2.783 ms with actual kernel overlap2.652 ms. This demonstrates real overlap after reducing consumer dispatch overhead; no production-video overlap claim.

### Active resources

One frozen InferHub job submitted (2 H-cluster GPUs, eight477 development videos): `zhouhe08__longlive_metadata477_development_Iter0__621b0701a202`. CPU prep passed; last observed waiting for GPUs after release back to queue. Do not resubmit it. Output root `/kaimm-distill/zhouhe08/longlive-system/outputs/sprint24h_metadata477_621b070`. Runtime commit621b070 is pushed; the in-flight job is unaffected by newer local changes.

Next frozen local video screen is ten153 development cases: Dense, Final, shared spatial grouping, conditional spatial grouping, and the same conditional route with the older executor, each on motion/state. All receive the same configured4GiB KV cache budget. Promotion requires whole-video evidence; do not claim the offline error improvement proves absolute video quality.
