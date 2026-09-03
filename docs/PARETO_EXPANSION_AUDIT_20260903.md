# Frozen Pareto expansion audit (2026-09-03)

Experiment identity: `fd197b7d66d27011432ee2315ffd10092544e812`.

The frozen expansion contained 102 cases: 12 matched RAG Dense references and
30 configurations for each of Fixed-K256, SCOPE-AR, and transfer-aware V-aware
history attention. All cases reached a terminal state on H200: 100 passed and
two SCOPE `recency_rank + per_chunk` cases failed with a preserved temporal
RoPE table-bound error. There were no missing cases or Dense fallbacks.

The successful set includes 16 videos with 240 latent / 957 decoded frames.
All successful videos passed full decode, latent shape/finite, artifact SHA,
backend, and fallback audits. A cache-continuous chunked VAE decoder fixed a
three-frame loss at the 120-latent chunk boundary; this post-generation harness
change is recorded separately from the experiment identity.

On four prompts by two seeds, including per-process model load:

| Method | SSIM | LPIPS | speedup vs RAG Dense | history transfer |
|---|---:|---:|---:|---:|
| Fixed-K256 | 0.650 | 0.326 | 1.59x | 25% |
| Transfer-aware V-aware | 0.647 | 0.323 | 2.68x | 25% |
| SCOPE-AR | 0.709 | 0.261 | 0.47x | 100% |

On the four 957-frame videos, the corresponding speedups were 1.20x, 1.97x,
and 0.37x. Full-frame diagnostics found no freezes, cuts, or flicker. Sampled
visual review found no subject/background reset or late collapse; it is an AI
visual audit rather than a blinded human panel.

Fixed-K256 and transfer-aware V-aware form the system Pareto. The latter is the
final speed/transfer candidate; Fixed-K256 is the quality-oriented guardrail.
SCOPE remains a quality diagnostic and negative system result because it is
slower than Dense and transfers essentially all candidate K/V.

The training gate returned `do_not_train`: the training-free method already
forms a useful Pareto point, and no post-hoc 50% density experiment is used to
trigger MSE/LoRA training.

Reproduction and analysis entry points:

- `scripts/build_pareto_suites.py`
- `scripts/build_pareto_partition_plan.py`
- `scripts/inferhub_batch_pareto_partition8.sh`
- `scripts/evaluate_formal_basic_quality.py`
- `scripts/build_case_metrics.py`
- `scripts/summarize_pareto_expansion.py`
- `scripts/plot_pareto_expansion.py`
