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

## 16:00–18:00: complete videos, representation probe and streaming VAE

- InferHub metadata477 completed8/8 on runtime-reported **NVIDIA H800**, not H200. All four paired routes/latents/rawRGB/noise/bytes exact. Full speedups Dense motion/state1.314×/1.360×, Final1.167×/1.112×. Recovered unchanged to `../../results/videos/sprint24h_20260907/metadata477_h800_v1`; source paths inside copied manifests remain original shared paths.
- Group153 completed10/10. Same conditional route under old/resident executors is bitwise equal, full speedups1.172× motion/1.147× state. Conditional grouping with resident execution is still slower than optimized Final:58.27 vs51.28 s motion;60.99 vs49.42 s state. Full latent relativeL2 versus Dense .261 vsFinal .298 motion; .238 vsFinal .231 state. These are fidelity, not absolute quality scores.
- Visual overview: all methods retain the red toy/blue ball and cup; no reliable absolute quality winner. Late toy motion is limited in all, and the cup fills early then mostly saturates. Conditional state retains a visible input tube in more late samples, but this is an observation, not a scored causal improvement. See `group153_review_v1` for boards and complete audits.
- Pure CPU group admission was optimized without changing selection: vectorized coordinate validation and certified block-score expansion replace repeated token-score sorting.24 CPU-reconstructed real-capture cases preserve routeSHA, ~2.88–3.58× faster complete selector+digest. Shuffled/noncanonical coordinates use the exact reference path. This is not yet a measured video speedup.
- Prototype-tail mechanism: represent omitted committed K/V by count-weighted post-RoPE block means instead of deleting them. Same legacy raw25% + tail reduces offline errors in all8 captured layer/category cases; random25% + tail is even better in these cases. BF16 rounding barely changes this observation. Proto-only is inadequate in middle layers. Block16 prototypes improve onBlock64 but do not eliminate middle-layer error. No online prototype-tail implementation or speed claim yet; FP32 and BF16 prototype storage/accounting are distinguished.
- Index-time residual/representation-aware raw admission is a next hypothesis, **not** a proven method. The score for dropping groups need not be the right score for deciding which groups require original KV when approximate summaries remain available.
- Streaming VAE module: main-thread decode dispatch on a separate CUDA stream, bounded2-slot pinned output pool, completion thread, optional non-collecting sink. Real saved-latent gate is bitwise exact versus batch decode:39latent/153pixels, batch9.011 s, stream8.379 s, first raw pixels .491 s. This is decode-only, not generation overlap.
- Stream-local timing fences are explicit, thread-local and opt-in; DMA readiness events are unchanged. Real Final full-forward/old-new executor gate passes. Added fragmented-MP4 incremental sink with canonical RGB hashing and first-packet timestamp; client display time is not measured.
- Next local factorial: true generation plus VAE plus incremental encoding, batch/async decode × device/current-stream fences. Same seed/route/latent/RGB checks; model load separate. Both raw first-pixel and first muxed-packet times are reported. Upstream detailed profiling is disabled for all arms; a separate representative timeline follows equivalence.
- Latest full CPU regression before this phase freeze:395 passed,1 skipped. Earlier test-fixture failures (legal dtype/element-size metadata and cross-test import) were fixed; they were not scientific negative results or production failures.

Push review temporarily required destination ownership evidence. Connected GitHub authenticated account and repository owner both Nanbuqaq/id100647184, with admin/push permission; unchanged push was subsequently approved. Evidence is stored in `results/infrastructure/github_sprint_push_ownership_20260907.json`. No destination or branch workaround was used.

## 18:00–21:00: model-level overlap, long group controls and representation

- Group477 completed10/10 technical pass on H-series GPUs. The collector failed on a redundant top-level latent-length field, not generation; corrected recovery uses canonical case identity. No videos regenerated. New/old grouped executors preserve all routes/latents/RGB, preliminary full speedups1.528× motion /1.283× state. These runs share host resources; isolated repeats are still needed for timing attribution.
- Nonblind sampled477 review: conditional spatial groups retain a more box-like red toy while Final drifts toward a humanoid/single-eye form. State videos all fill early, then saturate; no reliable state winner. This single seed does not establish quality promotion. Next registered controls: Final50/shared50 at the old seed and Dense/Final25/Final50/shared50/conditional25 at development seed20260908. These are higher-budget controls, not falsely labelled equal-byte comparisons.
- True generation→VAE→incremental fragmented-MP4 pipeline is implemented and tested. Eight153 factorial videos and six warmed priority-control videos preserve noise/routes/latents/RGB exactly. Warmed complete generation+decode+encode: Dense motion48.086→44.860→43.629s and Final state45.773→43.679→40.915s for batch / async / generation-high-priority async. First packet falls from45–47s to4.4–4.5s (priority5.3–5.5s). Client display latency is unmeasured. Independent timing replication is pending.
- Node-level full-model Nsight trace validates6.807s actual generation/VAE kernel overlap within41.654s parent wall time. VAE itself slows under contention; overlap is not free. Explicit D2D volume≈1.44TB is not a measurement of all HBM traffic. Facts: `streaming_priority153_v2` videos and `streaming_priority_trace_v1.activity.json`.
- Shared-prefix paged FA2 reference reduces physical packed KV215.65→103.81MB (unique original90.69MB), but resident3.150ms versus paged3.284ms is latency-negative. BF16 versus FP32 relativeL2≈.00230; output is not bitwise equal to resident. No production-video or original-H2D reduction claim.
- Prototype reference153 completed8/8, one causal generation per arm (not Dense two-pass). Added full-candidate moment materialization is charged: base H2D≈2.516GB plus≈10.575GB reference traffic per tail case; not online speed-Pareto eligible. Dense-relative full latentL2: motion Final .29827 / SDPA-null .30264 / tail64 .29147 / tail16 .28685; state .23054 / .24308 / .22559 / .22143. Backend-only drift is nonzero and remains a required control. Nonblind overview shows no convincing absolute task-quality gain; quartiles need continued inspection. See `prototype_reference153_review_v1`.
- Same-four-slot spatial/random/K-feature prototype comparison now passes both categories, layers0/9/19/29. K-feature has lower capture error in all8 cases than spatial/random grouping, but middle layers remain hard. Grouping uses completed K, not Q/teacher output. It is a representation probe, not proof of streaming video improvement or efficient index maintenance. Index-time residual-norm admission also remains mixed: layer9 fails to beat random/stratified raw choices.
- Research direction: with compressed summaries retained, raw admission should target information the summaries cannot express. Relevance under deletion need not rank residual value under a summary-plus-raw representation. Test this mechanism before expanding a new method claim.

## 21:00–21:30: controlled negative and two gated implementations

- All14 new477 budget controls completed on localRTX4090. New seed20260908 has a same-GPU five-arm comparison within each category. The old-seed missing50% arms are local but their reused controls are H-cluster: that subset is **cross-hardware diagnostic only**, not clean admission/timing attribution. Future suite states now include GPU model/capability and Torch/CUDA versions explicitly.
- New-seed conditional25 transfers26.73GB motion /24.63GB state, versus Final25 14.16GB, Final50 28.32GB, Dense56.65GB. Thus its physical traffic is47.18%/43.48% ofDense, not25%. Full times: motion Final25 182.67s /Final50 195.45s /conditional196.76s /Dense191.47s; state173.58 /198.10 /204.40 /188.57s. Single timings remain diagnostic, not statistically replicated speed estimates.
- Method-name-masked AI overviews were recorded before reading the mapping. Motion preferred the chunky shape retention ofFinal50 andDense; state preferred Final25's level retention. Conditional is not a stable winner. More Dense fidelity also did not imply better apparent state retention. No further broad conditional-spatial video promotion; retain as a negative grouping baseline. Exact observations and hardware caveat are in `group_budget477_review_v1/MASKED_FIRST_REVIEW.md`.
- Complete timeline copy attribution:1.340TB D2D /2.896s service belongs to `longlive/self_attention_complete`. Largest copy-size bin230,031,360bytes matches full12-frame FP64 RoPE output. Upstream does an often-empty suffix concatenation and then stacks FP64 buffers. New direct-output RoPE retains FP64-complex arithmetic and avoids those two intermediate copies; it does not silently switch toFP32.
- RoPE operator gates: four CUDA layouts including batch2, padding and strided inputs are **bitwise equal**; median operator speedups1.42–1.66×. Dense/Final real self-attention forwards preserve route and output exactly. These are not yet full-video speedups. Explicit config field `local_rope_layout=upstream|direct_output` enters system identity.
- `CommittedMomentRuntime` builds K-only groups and FP32 additive K/V sums once from already resident, completed/evicted frames; stores moments onCPU; onloads only small moments and coordinate labels in addition to the selected raw union; subtracts admitted raw contributions so no history is double-counted. Upstream-zero RoPE only. No full-candidate fallback; no bound on total CPU archive claimed.
- Real small/large forced-eviction gates pass spatial64, spatial16 andK-feature64×4. Full-candidate API is poisoned, four subsequent denoising cache hits are required, changed RoPE is rejected, evicted originalKV is exact, and weighted output passes isolatedFP32 gates (worst relativeL2≈.00317). Index compute/D2H, moment pack/H2D, subtraction and cache storage are recorded. Whole-video quality/time remains pending.
- Next local freeze:153-frame streaming×RoPE four-arm exact-output gate, plus two-category five-arm committed-moment effect/accounting videos. Conditional on those gates, one4-GPU H-cluster batch will run the same streaming×RoPE2×2 for Dense/Final and motion/state at477frames, three randomized blocked complete-trajectory repetitions per arm (48 total). Same-process timing repetitions are not independent-machine replications.
- CPU regression at this point:424passed,1skipped. Pure analysis remains onCPU; no H job from this phase submitted yet.

## 21:30–22:30: repeated H200 result and nonlinear-query insight

- Local153 streaming×RoPE gate completed4/4 with identical noise/routes/latents/RGB. Complete generation+decode+encode seconds: batch-upstream48.509, batch-direct44.646, async-upstream41.693, async-direct39.136. This is a single local pair/factorial, not an uncertainty estimate.
- Frozen4-GPU InferHub job `zhouhe08__longlive_streaming_rope_factorial_Iter0__1672f6b630d9` completed48/48 on **NVIDIA H200**, exit0. Recovered to `streaming_factorial477_h200_v1` with the original source paths preserved. Three blocked randomized repetitions for each2×2 arm, Dense/Final × motion/state. Every complete trajectory is bitwise equal within its method/prompt group.
- Median paired combined speedups: Dense-motion1.1615×, Final-motion1.1832×, Dense-state1.1596×, Final-state1.2028×. Variation is material: Dense-state has one combined negative repeat0.8967×. Several second-round runs are much slower across arms. Do not treat these as independent-machine replications or claim uniform improvement in every run.
- Direct RoPE alone under H200 async is median-negative for3/4 method/prompt groups (.956/.939/.924×), positive inFinal-state1.197×. Thus do not make it a universal H200 streaming default. The strong, consistent attribution claim remains unproven; further isolated diagnosis/repetitions are appropriate before hardware defaults are frozen.
- New node-level4090 trace preserves the external full trajectory. Explicit generation D2D bytes1.440TB→0.549TB and copy active time3.099→1.060s; VAE/generation simultaneous kernels6.807→6.781s. Profiled parent wall41.654→43.114s is **negative** across these separately collected traces; it is not the unprofiled speed estimate, and the eliminated copies do not alone establish whole-pipeline benefit. All HBM-transaction claims remain absent.
- Commit-time prototype videos153 completed10/10. Extra H2D for spatial16 falls10.575GB→1.289GB, with an additional1.010GB indexD2H. Motion full time49.633→49.404s is effectively unchanged; index+materialization1.921s versus full-candidate construction2.815s. Committed spatial16 differs from reference tail16 in full latentL2 .0961, so it fails the .01 latent equivalence gate and is treated as a changed numerical method, not a lossless rewrite.
- K-feature committed tail improves motion Dense-relative latentL2 .29827→.24474, but state .23054→.23119 is not an improvement. Spatial committed tail gives .28871/.22093. Nonblind overview retains both objects/cup; no convincing absolute-quality winner yet. `committed_moments153_review_v1` keeps complete/quarter diagnostics and additional index traffic.
- Seven moment-risk proxies plus legacy/random controls were tested with identical raw25% andK-feature tail slots. Spatial Q-centroid versions still fail state layer9 (best≈.1154 versuslegacy .11337); no proxy promoted from that screen.
- Key diagnostic: scoring **each current Q before averaging** instead of scoring averaged Q resolves that failure on the same isolated captures. Full-Q mass×projected-key-variance gives state layers0/9/19/29 relativeL2 .03814/.10207/.08717/.03038 versuslegacy .04237/.11337/.09972/.03416; motion .02931/.09830/.07046/.02749 versus .03371/.11565/.08474/.03537. No Dense output enters these scores. This is capture evidence, not a video result or a novel-method claim yet.
- Deterministic256-query representatives retain most of the state improvement but layer9 is≈2.9% worse than full-Q. Query-sampling accuracy/cost is still under study; no query-count parameter has been promoted to an online video selector. Full-Q GPU scoring uploads only compressed committed moments, not full candidateKV; its extra bytes and unwarmed diagnostic timing are reported separately.
- New development event workloads (canvas state / toy identity reveal-away-return) are frozen for Dense-only feasibility screening, with no claim they are valid benchmarks yet. Only active segment conditions reach generation, but known text prompts are preencoded and charged. Cross-attention invalidation only; this is not the official interactive self-KV recache policy.
- First21-latent event gate generated the reference video then failed post-hoc CPU/GPU latent comparison; v1 failure preserved. Fixed version saves latents before audit and compares onCPU. v2 passes reference/no-op events bitwise and exercises all three true condition transitions. No long Dense screen has run yet. CPU regression after the risk-interface additions:440passed,1skipped.
