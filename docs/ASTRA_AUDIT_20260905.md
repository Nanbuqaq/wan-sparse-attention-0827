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
