# System execution audit, 2026-09-05

Scope: review the in-progress system branch and preserve existing runs. The
complete plan remains conditional on numerical, runtime and quality gates.

## Confirmed findings

- **CPU oversubscription:** each local Dense process has 260 OS threads; the
  container CPU quota is 32 cores (3,200,000 / 100,000 us), despite an affinity
  mask exposing 256 CPUs. Completed screen cases report 4,579 / 4,963 seconds
  inside CPU gather. A bounded six-frame, BF16 real-capture CPU replay under
  these two running jobs measured 1.75 / 0.90 / 0.54 / 10.71 seconds with
  1 / 4 / 16 / 128 Torch threads. These are two-sample diagnostics under
  contention, not hardware or algorithm speedup estimates. Preserve the running
  screen; explicitly budget threads in all future lanes.
- **Incomplete transfer timing:** candidate concatenation occurred before the
  gather clock; GPU union reassembly and metadata work were also excluded from
  the reported gather+H2D sum. The new complete materialization clock includes
  these costs, with prepare, pack, allocation/pinning and device restore fields.
- **Strided source:** a fixed head in `[B,T,H,D]` has token stride `H*D`, not
  `D`. Existing direct-multirun results count Torch copy calls and infer payload
  bytes without proving physical CUDA transfers. Retract their use for layout
  promotion or effective PCIe bandwidth claims. The new archive-runs packer
  performs CPU packing into contiguous `[B,H,P,D]` and splits runs at distinct
  frame storage allocations. The full candidate tensor is not concatenated.
- **Case collision:** the two utility candidates differ only in method params,
  which were absent from identity v2. New system cases use v3 with the effective
  method params, and the ten-case builder now checks unique keys. Existing
  v1/v2 artifacts retain their original identity.
- **Batch result collection:** the new three video launchers expected
  `method_suite_states.json` while the actual runner emits `shard_N_states.json`.
  Their collectors are corrected before any of these video batches are launched.
  The artifact audit now compares backend to the canonical case key, including
  expected manifests that do not duplicate that field at the top level.
- **Baseline access to optimizations:** the prior Dense post-transfer branch
  bypassed the new TransferPlan/cache path. Dense now has a metadata-only full
  route that can use the same materializer and per-chunk cache as Fixed/Final.
- **Quality evidence:** the prior teacher used unrotated history-only Q/K,
  omitted sink/current/recent KV, and compared whole-Block64 candidates at about
  24.1% with a token-trimmed 25% legacy union. The candidate rankings remain
  screening information. Complete captures now save actual post-RoPE Q, exact
  K/V, full historical K/V, positions and the already-selected route. Matched
  physical-byte / block-granularity controls are required before promotion.
- **Unimplemented execution switches:** offload/onload overlap and dataflow
  fields exist in configuration, but the runtime still uses synchronizing
  copies and its sparse backend. They are not completed optimizations. The
  profile helpers are not yet a complete Nsys/NVTX timeline.
- **Execution replication:** grouped FA2 `_sequences` replicates exact/recent
  and selected history KV per query group, then concatenates these again.
  Top-p can multiply staging while reducing logical pairs at unchanged union.
  `_sequences` and final output restoration are outside the old backend clock.
  New detailed records expose complete backend wall time and estimated packed
  KV storage separately; the latter is not a measured HBM transaction count.
  The Python `kvout_online_reference` loops over query groups outside KV blocks;
  it is an online-softmax correctness reference, not a demonstrated KV-stationary
  execution. Thus the old partial Attention fraction cannot alone stop or
  promote KVOut before complete backend and pipeline profiling.
- **Cache coverage:** checksum hashing previously failed for BF16 NumPy
  conversion; raw block cache keys omitted batch. Both are corrected. Cross-chunk
  raw cache still scatters tokens individually and is an unpromoted prototype.

## Evidence priorities

1. Run complete CUDA self-attention forwards for Dense, Fixed, Final and utility,
   comparing source gather / packed runs / per-chunk cache with identical weights,
   routes, RoPE, and original K/V. Test all newly used branches before video batches.
2. Establish an end-to-end video benefit against comparably implemented baselines,
   with all CPU, copy, plan-building and GPU reassembly included.
3. Revisit value admission using complete post-RoPE captures and matched byte
   budgets. Keep the existing failed cost model negative. A new model would
   require a new isolated calibration and held-out set, plus decision-regret
   and ranking audits, not just a refit on the old holdout.

Artifacts under `results/metrics/astra_audit/` are diagnostics. No new route,
layout, cache or kernel is promoted by this source audit alone.

## Follow-through at the continuation checkpoint

- The four state-screen videos are technically complete and storyboard-reviewed.
  `system_holdout_prompts.json` is frozen at SHA-256
  `6d898e96dd28e924622d5585ab6cc85b139560282c01761cde87b518f3d17fea`.
  Blue canvas is eligible, but its early plateau limits evidence for continuous
  state updates. Review provenance is assistant visual review, not a human study.
- Source `5259c04` real-CUDA synthetic full-forward regression on RTX 4090:
  five-call medians at Q=4680/history=9360 are Dense 0.81825 -> 0.16322 s and
  Final 0.15511 -> 0.11609 s (legacy -> archive runs + cache). Max-abs difference
  is zero and routes are identical. These are not full-video speedups.
- Source `687d014` replaces redundant static-admission argmax scans with one
  stable sort, checked against an independent scalar selector (30 random cases).
  Peak-utility GPU regression passes all four materializers; legacy/cache
  five-call medians are 0.29537 / 0.25038 s, with identical routes and outputs.
- Count-uniform + Top-p 0.95 also passes, but exposes **47.054x estimated KV
  staging replication**: 3,376,697,344 packed bytes vs 71,761,920 unique resident
  bytes; 677 active groups; peak allocated memory about 7.53 GB. Five-call
  legacy/cache times are 1.10547 / 1.07222 s. Candidate differs from peak, so
  these two rows do not establish a causal Top-p speed ratio. This is a strong
  reason for a same-route executor timeline, not proof of measured HBM traffic.
- Dense-only four-lane video validation is already queued at source `a473e34`.
  The ten-case routing protocol reserves its legacy Dense motion case with that
  actual source SHA; only nine new cases are scheduled. Final lanes additionally
  collect isolated post-RoPE complete Attention snapshots with raw routing Q/K,
  exact KV, frame geometry and selected route. Their capture-augmented wall time
  must not be used for speed promotion.
- Opt-in NVTX scopes now cover complete self-attention, plan/cache/materialize,
  full-candidate concatenation, CPU route, group KV replication and restoration.
  Bounded cudaProfilerApi replay excludes startup and warmup. These remain
  nested launch scopes, not additive service-time or overlap evidence.
