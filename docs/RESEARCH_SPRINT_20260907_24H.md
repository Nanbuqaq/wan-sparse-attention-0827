# 24-hour research sprint: grouping, lifetime, and executable structure

Started: 2026-09-07 14:21 UTC. Discussion horizon: approximately 2026-09-08 14:21 UTC.
Starting commit: `0af48f290fad1b0c0d98b0d853d8fe6f66db4f65` (verify exact SHA in run manifests).

## Objective and boundaries

The objective is a credible faster-and-better streaming long-video method, not a complete Tether reproduction or a positive result for every kernel. The three research questions are: which information groups matter to which queries; how their structure enables sparse execution; and how their retention/update policies evolve during streaming.

Existing 44/102-case results and first-round formal results remain read-only. No training, shared-environment changes, repeated completed matrices, or new claim that consumed formal prompts remain unseen. Tether expansion stops: existing source-weight-matched long videos are mechanism evidence, including negative results. GPU0/GPU1 are available; InferHub GPU-heavy stages require current platform instructions, real-GPU branch gates, tests/dry-run/push, and one frozen batch per stage. Report actual GPU model, not requested model.

## Schedule and promotion gates

| Elapsed | Work | Required evidence / decision |
|---|---|---|
| 0–4 h | CPU route-control breakdown; exact-safe descriptor reuse; resident group metadata; same-route backend preparation | Measure full preparation, not only kernel. Retain disabled controls. Mutation/cache invalidation tests and real CUDA numerical gates precede video. |
| 2–8 h | Grouping development captures: random/spatial/feature or information-role partitions; query-conditional allocation and age/state interventions | Equal actual bytes and scheduled pairs where attributing grouping. Full exact/current/recent KV in teacher. Offline evidence never enters current online selector. |
| 4–12 h | Real producer/lookahead pipeline: CPU pack → H2D → group consumption, finite buffers and same logical edges | Include archive layout/pack, pinned waits, producer startup/amortization, missing/extra bytes, peak memory. Nsight must prove useful overlap and lower full latency. |
| 8–18 h | Model integration and development 153/477 video checks; conditional 957 or H-series GPU confirmation | Dense and sparse receive common execution improvements. Same-route changes require exact latent/output or explicit numerical gate; changed routes require absolute task/late-quarter review, not Dense fidelity alone. |
| 18–22 h | Conditional new held-out long videos; 2×2 method/system attribution | Freeze new prompts/seeds and selection before viewing candidate outputs. No broad expansion if development gates fail. |
| 22–24 h | Terminal audit, figures, evidence tables and next-discussion synthesis | Every launched case pass/fail/negative; preserve failures. Separate implementation defects, hardware limits and scientific negative results. |

These are time allocations, not promises of positive results. Move effort when evidence rejects a hypothesis; do not spend GPUs on pure CPU routing.

## Immediate hypotheses

1. **Control-plane reuse:** KV cache hits remain expensive because coordinates, positions and SHA identities are rebuilt before lookup. Reuse must preserve exact cache correctness, including in-place/alias mutations and inference tensors without version counters; never weaken keys to gain speed.
2. **Structured consumption:** query-group metadata and repeated KV materialization can dominate Attention execution. Keep the same group-to-history graph while changing execution. Separate a native full-history RAG path from an adapter-heavy reference when making baseline claims.
3. **Actual supply pipeline:** two streams alone are insufficient. Produce future pages ahead of consumption, enforce host/device slot ownership with completion events, and measure the whole critical path.
4. **Information lifetime:** stable identity, changing state and revisitable scene can require different history. Do not assume mutually exclusive roles or monotonic benefit from more old memory. First show an equal-budget intervention before building a complicated online policy.

## Accounting

Model loading is a separate cold-start task (the observed ~165 s in four `torch.load` calls is not pure PCIe time). GPU service, host ranges and exposed waits remain separate. Full-flow instrumentation has overhead; steady-state comparisons require disabled diagnostic controls. Report logical history pairs, scheduled pairs, unique transferred KV, metadata traffic and physical padding separately. The 25% union is a reference point, not a constraint on new exploratory policies.

## Initial state

Both local RTX4090s idle at startup; no existing research process stopped. Preserve untracked `scripts/audit_retrieval_divergence.py`. No new GPU batch submitted yet. Production overlap is still explicitly unimplemented; no asynchronous video speedup is claimed.
