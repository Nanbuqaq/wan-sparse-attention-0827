# Research insight cards — evidence, limits, next discriminator

These are development findings, not claims that a new method is already a general faster-and-better winner. Exact source details remain in the linked reports and result manifests.

## 1. Fewer bytes can expose a different critical path

- Same FP64 RoPE, direct output: local operator1.42–1.66×, exact outputs; explicit generation D2D1.44TB→.549TB.
- Complete H200 streaming×RoPE factorial48/48 exact trajectories; median combined1.16–1.20×, one negative combined repeat and noisy component attribution.
- DirectRoPE under H200 async is not consistently beneficial. A byte reduction alone is not a uniform end-to-end result.
- Motivation: characterize data movement and exposed waiting at the whole-pipeline level, then choose hardware-specific execution.
- Evidence: `streaming_factorial477_h200_v1/factorial_audit.json`, `direct_rope_stream_trace_v1.activity.json`, `SPRINT24H_PROGRESS_20260907.md`.

## 2. The order of query aggregation matters

- Spatial Q means before nonlinear scoring fail the difficult state layer9 admission screen.
- Score actual current queries against committed prototype statistics, then aggregate: full-Q mass/variance improves all8 complete captures.
-1024 deterministic query representatives remain within2% of full-Q output error;256 misses that criterion in state layer9.
- Motivation: small summaries must preserve the relevant query distribution, not only its mean.
- Limit: this is a capture result. The resulting wire method improves motion fidelity but not state fidelity versusFinal in153 videos.
- Evidence: `moment_risk_*`, `precision_wire153_review_v1`, `PRECISION_QUERY_DISTRIBUTION_20260908.md`.

## 3. Compression value is different from deletion value

- Count-weighted K/V prototypes can preserve information lost by dropping omitted blocks. Raw admission should target information that the summary cannot express.
- WholeBlock64 raw14% plus compact wire and full metadata measures≈22.93% H2D versus complete first-route candidates; numerical and original-KV gates pass.
- But representation-only477 tails are slower and do not establish quality improvements; some show substantial identity redesign or state regression.
- Motivation: jointly test representation, admission and memory influence rather than assuming lower Dense error is the final objective.
- Limit: original CPU KV is still retained, so this is not yet bounded-memory compression.
- Evidence: `precision_runtime_all_admissions_large_v3.json`, `committed_moments477_review_v1`, `prototype_wire.py`.

## 4. Timely memory access can matter more than persistent exposure

- First-return-only privileged anchoring improves the original identity family in two seeds at unchanged total raw-history bytes and identical pre-intervention latents.
- Default retrieval reaches relevant history too late in one seed and never reaches it in the other.
- Persistent anchoring is not monotonically better: one return undergoes a blue-body transformation.
- Motivation: memory activation/deactivation and event timing are research variables, not just a per-call Top-k score.
- Limit: the anchor interval is privileged and the workload is an absence/reappearance stress, not a fully successful natural camera pan.
- Evidence: `episode_anchor477_review_v1/DESCRIPTIVE_REVIEW.md`.

## 5. Correct past-episode ranking is not retrieval intent

- Current-minus-previous text-key cosine selects the old reveal episode where ordinary cosine selects away context.
- Automatic477 result is mixed: one seed retains the identity family, another later drifts.
- The explicit no-robot control fails: contrast retrieves the reveal episode and injects a conspicuous robot. Dense keeps the foreground robot absent.
- Motivation: restore, replace and remove must be separated; remembering the right old object can still be the wrong action.
- Limit: no simple contrast-rule promotion; any intent fix needs independent negative controls, not a patch evaluated only on this failure.
- Evidence: `event_retrieval477_review_v1/INTERPRETATION.md`, frozen `event_retrieval_negative_controls.json`.

## 6. More resident context can impede a requested transition

- Cache12 is not dense recent12 in the RAG allocation. At latent78 exact frames are0 and76–80; coarse eligibility≤61, leaving guaranteed gap62–75.
- Full local12 avoids the history archive/onload and runs the four477 controls in≈105–108s on4090, but does not cleanly satisfy the absence/reappearance instructions in both seeds.
- One apparent identity retention is actually failure to leave view; the other generates a new large subject during the away interval.
- Motivation: compare response to new instructions, preservation of old identity, and memory cost jointly.
- Limit: cross-only switching is not the official interactive recache protocol. All8 source-locked native controls now complete; recache also fails clean original-identity restoration and negative controls. The workload remains a development stress, not a successful natural camera-pan benchmark.
- Evidence: `recent_context_geometry_v1.json`, `local_only_all_review_v1`, `RECENT_CONTEXT_ALLOCATION_20260908.md`.

## 7. A visual frame does not uniquely identify a KV representation

- Non-mutating official recache observer passes4/4 full noise/latent/RGB identities.
- Same completed frames at switch30: layer9 K/V relativeL2 .3274/.5640; layer19 .3496/.6009. Both condition and recomputation context differ.
- Motivation: which version is committed to long-term memory may matter in addition to which frame is retained.
- A bounded6-frame pre/post snapshot intervention passes3/3 GPU39 gates with exact pre-return prefixes; its two-seed477 comparison is running, not yet a quality result.
- Limit: version differences are expected, not novelty by themselves. Episode capture/return boundaries are privileged; this probe is not autonomous retrieval.
- Evidence: `recache_version_capture39_v1`, `episode_snapshot39_gate_v1`, `EPISODIC_KV_VERSION_PROBE_20260908.md`.

## 8. Region influence is not whole-video quality

- Existing source-weight-matched Tether477: teapot0 gains spurious hands/light arcs; teapot1 loses the required centered subject. No stable overall quality winner.
- Both automatic cyclist masks fail; manual initialization rescues one seed only, the other still fails tracking. These are explicit role-acquisition limits.
- Neutral zero-bias split-SDPA also changes long trajectories and retrieval: do not assign every difference to semantic bias.
- Motivation: separate group representation, activation, update/lifetime and on-policy alignment; assess subject/scene/task jointly.
- Limit: fixed-mask misalignment is a hypothesis, not an isolated causal explanation. No new fair AdaCluster/SVOO/SCOPE ranking exists.
- Evidence: `TETHER_LONG_VIDEO_FINDINGS_20260908.md`, original video/mask terminals and newly completed descriptive boards.

## Remaining paper-level requirements

- Reliable long-video absolute quality, independent schedules/seeds and negative controls; no universal algorithm winner yet.
- Strong matched baselines including correct prompt-switch behavior; LongLive2 is source-audited, not fairly measured against this1.3B backbone.
- Bounded CPU history and end-to-end steady-state behavior remain unresolved for the history methods.
- Preserve all negative results and separate component, trajectory-equivalence, and semantic-quality claims.
