# Conditional approximate VAE layout characterization

The completed numerical gate rejects all changed layouts from the lossless path.
It does NOT establish whether the numerical/performance tradeoff is worth further
study. A bounded decoder-only replay will measure that tradeoff without changing
the active generator/pipeline or creating new DiT videos.

Compare baseline and weights_only, chosen before timing because it needs no
per-convolution Python/input-format hooks; the other two variants share its RGB
hash on the numerical prefix but are not assumed to have measured equal speed.
Saved native16-latent prefix/61pixels, same source and device; first-call, five
warmups per mode and30 seeded randomized pairs. Gate RGB must match its own
recorded variant before timing. Preserve BF16-vs-BF16 error, not an FP32 claim.

Measure synchronized reset+decoder+native float/clamp wall and CUDA stream spans.
No CPU error/hash, input H2D, layout setup, encoding or DiT inside this timer.
It is an isolated approximate-decoder regime, NOT an adopted system optimization.
Define latency reduction as1-minus(layout median / baseline median).
If median decode reduction is below10%, stop this branch for now. If >=10%, only a
full saved-latent output/temporal review can qualify further approximate study;
do not turn a speed result into quality acceptance or reuse exact-pipeline claims.
