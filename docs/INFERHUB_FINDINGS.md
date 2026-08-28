# InferHub GPU findings

This file records reusable implementation findings only. Large logs, videos,
weights and internal paths are intentionally excluded from the public repo.

## H200 Triton control-flow failure

- A first rectangular varlen attempt used a data-dependent Triton `while`.
- The H200 Triton compiler aborted in `FenceInsertionPass` because that pass did
  not support `WhileOp` in this kernel.
- The implementation now uses a compile-time `MAX_K` and a static
  `for range(0, MAX_K, BLOCK_N)` loop with per-sequence masks.
- Backend gates run in separate processes so a compiler abort is preserved as
  a kernel-negative result and cannot erase routing, grouped-FA2 or fixed-kernel
  evidence.

## Batch policy

- A multi-GPU batch assigns different work to every requested GPU.
- Dense baselines are not repeated unless a changed measurement path or a new
  sequence length is being validated.
- Route plans are serialized once and replayed unchanged across grouped FA2,
  fixed rectangular Triton and varlen rectangular Triton. The route-plan SHA is
  recorded in every backend result.

## Post-transfer execution and density accounting

- Post-transfer methods first move the complete coarse-retrieved candidate KV
  set, then map the route-plan coordinates back into that dense transferred
  order before executing the selected Q-K graph.
- `history_pair_density` is accumulated from actual per-query/per-head Q-K
  pairs. It is not derived from unique transferred tokens.
- `history_transfer_density` uses actual transferred K/V bytes divided by the
  complete candidate K/V bytes, including staging padding when present.
- Method-specific LongLive calibration values are explicit `method_params` in
  the run config and cannot alter method identity or routing stage.

## Stage-batch recovery findings

- When an InferHub task declares a project root as `INFER_WEIGHTS_DIR` so it can
  read both the model bundle and prior QKV captures, `inferhub_entry.sh` must use
  an explicit `LONGLIVE_INPUT_BUNDLE_ROOT`; otherwise model/checkpoint paths are
  resolved one directory too high.
- Optional capture discovery under `set -e -o pipefail` uses an explicit
  failure-tolerant lookup before falling back to a previous verified capture.
- A zero-history route such as RAG Local bypasses history RoPE/materialization
  and replays the same backend with an empty history tensor plus exact KV.
- Successful load-once cases are reusable only after video SHA, full decoded
  frame count and latent artifact verification. Failed cases are retried.
- Query groups with identical selected-history sets are compacted in the route
  plan. This preserves every Q-K pair while reducing grouped/rectangular kernel
  packing overhead.

## Calibration and low-utilization recovery

- Calibration route plans serialize coordinates on CPU, while captured Q/K may
  be evaluated on CPU or CUDA. Coordinate lookup explicitly moves plan fields
  to the candidate tensor device and sorts encoded coordinates before
  `searchsorted`; this also handles unordered frame/token captures.
- A two-GPU sequential completion batch can be killed by InferHub's low-GPU-
  utilization guard when one lane finishes early and the other performs several
  model loads plus calibration. Retrying successful videos is unnecessary.
- The recovery batch performs real-QKV calibration in the no-GPU prep phase,
  then assigns matched Dense/100%-route correctness and each of the four paper
  methods to five distinct GPU lanes. This keeps the frozen stage batched while
  avoiding idle cards and preserves every previously verified artifact.
- A 39-latent Dense run cannot be sliced into a valid 21-latent correctness
  reference: allocating a longer initial noise tensor changes subsequent RNG
  consumption. The 100% latent gate therefore uses separately seeded,
  equal-length 21-latent Dense and sparse runs from the same commit.
