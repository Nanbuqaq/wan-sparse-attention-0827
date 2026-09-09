# Pixel completion: full equality and a matched two-GPU latency signal

Frozen runtime8f49008, same native BF16/FA2 model, same two4090s, same inputs,
same qualified strict-checkpoint constructor on both arms. No decoder layout or
floating-point operation is changed. Both253 and509 gates pass complete actual
latent/decoded RGB, byte accounting and per-buffer reuse-after-sink checks.

| Full509, one matched pair | Inline CPU sink | Bounded completion thread |
| --- | ---: | ---: |
| Generation-start to complete delivery | 98.804888s | 84.594745s |
| First muxed packet | 4.733633s | 4.842218s |
| DiT producer backpressure | 7.488363s | 0.000215s |
| Pinned input | 5.407MB | 5.407MB |
| Pinned pixel output | 43.254MB | 86.508MB |
| GPU1 peak allocated | 12.133GB | 12.133GB |

Observed complete-delivery reduction~14.4%, with no first-packet improvement
in this pair. Raw transfer bytes remain identical. Output buffering stays below
the explicit128MiB host-pin budget; the extra43.254MB is recorded, not hidden.
Short control was8.670→7.212s, useful for correctness but not a long-shape claim.

These are single-pair gates, not repeated speedup statistics. CPU thread
separation is not GPU overlap proof. Next: an equivalent native Nsight capture
and repeated paired timings; preserve load/first-packet/resource accounting.
No one-GPU throughput superiority is claimed: that baseline must receive the
same output-stage optimization before comparing resource efficiency.

Facts: `results/metrics/memory_activation_20260910/native_pixel_completion_gate_v1/`
short.json/full.json. Both gate sessions29116/5111 are closed; do not rerun them.
