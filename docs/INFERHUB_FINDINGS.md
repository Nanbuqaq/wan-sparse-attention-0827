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
