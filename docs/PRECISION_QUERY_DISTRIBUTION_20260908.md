# Development candidate: query-distribution-aware whole-block precision

This is a research candidate, not a proven faster-and-better method or novelty claim. The previous representation-only477 screen did not establish a stable quality gain.

## Mechanism and causal boundary

For each completed frame, each head and each within-frameBlock64, build four K-feature groups using fixed four-iteration K-means. Store BF16 K/V means, int16 multiplicities, and independently scaled uint8 diagonal K variance. The variance is routing metadata, not an extra attention key. Original archived KV remains untouched onCPU; total CPU history is still unbounded.

The selector samples1024 uniformly spaced **actual current post-RoPE queries**, computes nonlinear scores per representative, then averages:

\[
a_{q,p}=\operatorname{softmax}_p(q^T\mu_{K,p}/\sqrt d+\log n_p),\qquad
r_b=|Q_s|^{-1}\sum_{q\in Q_s}\sum_{p\in b}a_{q,p}\,(q^2)^T\operatorname{diagVar}(K_p)/d.
\]

This is a heuristic, not a guaranteed error bound. It never reads full unselected candidateKV, Dense output/residual, or future generated frames. Unlike the failed spatial-centroid proxy, averaging occurs **after** nonlinear scoring. On eight complete development captures,1024-query mass-variance selection stays within2% of full-Q output error; that calibration does not establish video quality.

Rank by risk per known raw-KV byte. Select whole blocks up to14% raw tokens per head, including the actual24-token frame-tail blocks. Do not split blocks. Every non-raw block remains as four count-weighted prototype nodes; every raw block removes all its own prototype nodes, avoiding double counting and rounded-sum subtraction. Raw head padding and inactive prototype slots are masked explicitly.

`HistoryRoutePlan` carries raw coordinates plus the representation recipe and whole raw-block IDs. Complementary virtual nodes are declared in metadata and are not mislabeled as exact original-token edges. Report raw density, represented coverage, virtual attention pairs, physical padding, and total traffic separately. This changes the raw-admission objective, so the old70/15/15 deletion budget is a control, not silently claimed as preserved.

## Wire, accounting and gates

Wire fields are BF16 K mean, BF16 V mean, uint8 variance, FP32 variance scale and int16 count. Full measured history H2D also includes rawKV, restore indices/valid masks, RoPE positions, and prototype control masks/counts. The active exact-compact materializer previously left restore/position byte counters atzero; new counters expose them without redefining old KV-only density measurements. Existing Nsight traces already include the actual copies and remain valid.

RealGPU small and full LongLive-shape gates verify: no full-candidate access; frozen five-call cache reuse; original evictedKV unchanged; exact raw/prototype coverage; and BF16 versus an independent same-representation FP32 operator. The large gate measures22.93% full history H2D relative to first-route full candidates, below25%. This is a component correctness/budget gate, **not** a video speedup.

The method is registered as `whole_block_precision_history` and requires an explicit `WholeBlockPrecisionRuntime`. Missing installation raises an error. Unsupported RoPE/refresh/backend settings are rejected. Ordinary Dense/Final receive common system improvements; no hidden Dense fallback or future teacher is used.

## Frozen next10-case screen

Two existing development categories (`calibration_motion`, `calibration_state`), seed20260904,39latent/153pixel frames. Five arms each:

1. legacyFinal25;
2. SDPA-null25 (backend-only numerical control);
3. wire+whole random14;
4. wire+mass/value14;
5. wire+mass/variance14.

All use the same exact-compact, cached archive path and direct-output FP64 RoPE on local4090. The random/mass arms keep the same wire/index implementation to isolate admission; they are **not** advertised as separately optimized baselines. Index construction/D2H, prototype onload, GPU scoring, CPU planning, raw onload, attention, VAE and encoding are charged.

Promotion requires both-category whole-video evidence, not only Dense fidelity or the capture result. Full history H2D must remain<=25% including the declared metadata. This is development, not an untouched formal holdout. Negative cases and the SDPA numerical-control divergence remain visible.
