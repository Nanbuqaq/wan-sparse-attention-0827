# Live continuation checkpoint (2026-09-08 03:05 UTC)

Continue the active goal; do not stop or resubmit running work. Discussion horizon remains2026-09-08 14:21UTC (~11h16m left). User wants faster-and-better causal streaming long video and research insights, not a perfect Tether reproduction. No subagents unless explicitly authorized. Old results are read-only; no training.

## Current source and resources

- Main repo `/home/zhouhe08/MyProjects/0904-longlive-system/publish_repo`, branch`longlive-system`, HEAD pushed`ae5d6d85b5e2018658343aa7a1d7d3d04faff810`.
- Preserve pre-existing untracked`scripts/audit_retrieval_divergence.py`. Last checked other work clean.
- LocalGPU0/GPU1 both4090, currently no remaining local generation job known. Use physical locks via`run_on_free_gpu.py`; sandbox GPU visibility is misleading, real GPU tools require escalation.
- One running frozen4-GPU InferHub job: `zhouhe08__longlive_event_retrieval_controls_Iter0__ae5d6d85b5e2`.
- Remote output `/kaimm-distill/zhouhe08/longlive-system/outputs/sprint24h_event_retrieval_ae5d6d8`; code/prep succeeded; at03:05 no lane progress summary yet. Inspect lane logs/read-only status, preserve job and do not duplicate submission.
- Submission script `../scripts/submit_sprint_event_retrieval.sh`; source/current manual and `/kaimm-distill/infer_hub/SKILL.md` were read completely. Public environment remains untouched. Each real resource phase needs GPU branch gates, tests/dry-run/push and one frozen batch.
- Active goal exists; do not mark complete merely on reaching a milestone. Latest full CPU regression459passed,1skipped, plus4 new event-collector tests passed.

## Most important completed findings

1. **System**: `streaming_factorial477_h200_v1` has48/48 actualH200 videos, three blocked randomized repetitions of streaming×direct-outputRoPE forDense/Final × motion/state. All paired routes/latents/RGB exact. Median combined speedups1.16–1.20×, but variation is large; one Dense-state combined repeat0.8967×. DirectRoPE under H200 async is negative for3/4 groups, so no universal default. Recovered2.1GB unchanged; do not repeat this matrix.
2. **RoPE**: same FP64 arithmetic, avoid FP64 cat/stack. Local exact output gates1.42–1.66× operator speedups; representative explicit D2D1.44TB→.549TB, not all HBM. Profiled parent wall itself was negative across separately collected traces; use unprofiled comparisons, not copy-byte reduction alone.
3. **Grouping**: conditional spatial raw25 had43–47%Dense H2D in new-seed477 and did not beat high-budgetFinal50 reliably. No further promotion. Old-seed reused controls mixedH-cluster/local hardware and are diagnostic only.
4. **Representation-only**: committed K-feature/spatial tails plus oldFinal raw25 completed8 new477 videos, slower and no stable quality gain. K-feature motion redesigns the toy; both state tails show level reductions. `committed_moments477_review_v1` holds boards. Do not promote from earlier153 fidelity gains.
5. **Query distribution**: mean-Q-before-softmax risk proxies fail state layer9. Scoring current Q separately then averaging improves all8 complete captures.1024 uniformly sampled queries are within2% of full-Q error (worst≈1.79%);256 state layer9 missed that gate. Uses only Q and committed prototype/moment statistics; no Dense output in selector.
6. **Compact precision**: registered`whole_block_precision_history`, raw wholeBlock64≈14%, four BF16 K/V means per block, uint8 scaled K variance +int16 counts. All unselected blocks remain virtual prototype nodes. Full large-shape runtime H2D including raw/wire/restore/position/control≈22.93%, original archiveKV unchanged,5-call cache4hits, BF16-vs-independent-FP32≈.0029. Files`precision_runtime.py`, `prototype_wire.py`, `moment_risk.py`. Explicit runtime mandatory; no fallback. Total CPU archive remains unbounded.
7. **Precision10-case153** complete: Final/SDPA-null/random-wire/mass-wire/risk-wire × two categories. Risk motionL2 .23209 versusFinal .29827, but state .23825 versusFinal .23054: **fails both-category Final-fidelity promotion**. Single full times motion48.354→42.561s, state46.752→42.770s; first-arm/warmup effects remain. Full historyH2D≈22.9%, versusFinal26.61% including newly counted metadata (25% is rawKV-only). No formal co-design winner.
8. **Recent allocation**: actual config is cache12/sink1/history6/chunk3/recent-exclude5. Atcurrent78 exact frames are0 and76–80; coarse eligibility≤61. Frames62–75 excluded from both paths:5 excludedCPU,3 newly evicted after coarse selection,6 unused-but-GPU-resident. This is allocation, not a demonstrated bug. `recent_exact_frames=3` must not be described as3 completed previous frames. Script/docs audit it.
9. **Local window8**: optional method parameter`exact_local_window_frames=8`, offered toFinal andwire equally. Eight153 cases completed under`local_window8_precision153_v1`. Gates exact across old/resident executors. Visuals do not establish a winner; motion turns toy away, state broadly saturates. Against original-windowDense, fidelity naturally diverges; do not call that same-context error. No long promotion yet.

## Key causal identity insight and live automatic candidate

- New Dense event workloads4/4 onH200 recovered`revisit_dense477_h_v1`. Canvas painted but never left view, so invalid forgetting benchmark. Toy disappears/returns in both seeds, but not a clean physical camera-pan; label absence/reappearance stress.
- Default returns a different rounded white-face toy. At first return chunk78 both use away-scene memory. Seed10 never retrieves reveal episode; seed09 retrieves it only at81, after changed identity already appeared.
- Privileged `EpisodeAnchorProbe`: two seeds × Dense / first-return-only / persistent-old-anchor,6/6 local477 complete. All pre78 latent prefixes exact; all raw history bytes56,645,222,400. First-return-only strongly preserves square head/black panel/triangle family both seeds. Not perfect: blue lower parts/extra yellow ball. Persistent seed09 has blue-body distortion. Evidence supports timing, not “more old memory always better”.
- Facts/boards: `episode_anchor477_review_v1`, especially`DESCRIPTIVE_REVIEW.md`. This is a privileged past-interval diagnostic, not automatic online/Pareto.
- Existing UMT5 text-key probe: plain cosine prefers away episode; current-minus-previous cosine selects earlier reveal/changed-state episode in both known input schedules. Text-only evidence, not generalization.
- Implemented `EventContrastRetrieval`: detects current condition-object changes, stores only past episode keys/closed time intervals, mean-pools current condition inFP32, chooses past episode by contrast (or cosine control), then its last6 **already archived** frames. Only first chunk after event, after≥2 prior closed episodes; ordinary visualRAG otherwise. No workload roles or manual anchor intervals passed. No future condition objects read. It may misretrieve on new-topic instructions; episode table/CPU archive still unbounded.
- Real39-latent `event_retrieval_gate39_v1` passed3 arms. Contrast chose past reveal6–11 and exactly reproduced first-anchor gate output. Cosine chose away13–18. Prefix checks pass. Gated generator code unchanged in latest batch-only commit.
- Live H batch is frozen10case477: original toy2seeds×Dense/cosine/contrast=6; duck new-topic seed09×Dense/contrast=2; explicit absence seed10×Dense/contrast=2. Negative controls were frozen before automatic videos in`event_retrieval_negative_controls.json`. Evaluate wrong old-object injection, identity and instruction following before promotion. Do not assume contrast is a winner.

## Next useful work

1. Monitor/recover live automatic10case batch; audit all technical states and raw-byte/prefix equality; phase/return storyboards for original and negative controls. Do not resubmit.
2. If automatic candidate works, test independent schedules/seeds and stronger controls before making paper claims. Its current trigger also operates on away transitions, so compare prefix only before first actual override, not always before return78. Frame metadata is causal but not a general semantic-memory solution.
3. A possible systems follow-up is bounded/on-demand episodic retrieval (currently **not implemented**); only investigate if timing/identity evidence supports it. A possible precision follow-up is summary-mass versus recent-context allocation (also not implemented as a joint method); current recipe failed state-fidelity gate.
4. Strong fixed-local baseline and nativeLongLive2 fair comparison remain open. LongLive2 source audited only; different5B backbone/precision/hardware means no “beat LongLive2” claim.
5. Refresh figures and concise insight/ablation report; current`insight_figures_v1` contains3 source-hashed figures, not the latest event results. Update progress after midnight (main progress file currently only through01:45).
6. Finish terminal audit, figures/video index, code/config/log/SHA handoff by14:21UTC, retaining all failed/negative cases. Do not force a positive method result.

## Known bookkeeping details

- `prototype_cases` reviewer count canbe4 or5 per category via`--expected-variants`; new review includes method config and full declared H2D. It explicitly says Dense fidelity is not same-context equivalence.
- First prototype reviewv1 sliced10 instead of9 late latents; erratum preserved, correctedv2 available.
- First revisit event gate generated video but failed CPU/GPU error comparison; v2 fixed and preserves latents before audit. Failedv1 stays.
- Metadata counter correction affects exact-compact restore-index/valid-mask and RoPE copies; old KV-only percentages not retroactively all-transfer percentages. Nsight actual memcpy totals remain valid.
- Keep source/runtime/GPU names distinct: earlier jobs H800; repeated477 factorial and Dense revisit screen actualH200; local4090. Never pool absolute times across them.
- Memory registry used this turn for IO/causal audit conventions: MEMORY.md35–39. Final response needs the required single memory citation block; no memory files were modified.
