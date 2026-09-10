# Duration-only native baseline probe

This separates elapsed generation/away length from candidate K and archive
growth. Native32/global8/pin8/block8 stays fixed. Toy/bead first source remains
at the original time; only the away interval grows, then the original return
segment is shifted later. There are still three scene closures, so raw scene
archive count does not grow. These are scripted newly generated histories,
not a natural user access distribution or repeated copies of a stored KV bank.

At24fps, registered full-resolution lengths are128/184/728/3608 latent frames:
509/733/2909/14429 pixels, about21.21/30.54/121.21/601.21seconds. Small technical
gates use64 or96 latent frames. Keep strong native and whole-source causal scene
memory controls; the unsuccessful partial-source candidates are not promoted.
Other axes (archive count/capacity, actual K, query chunk) require separate tests.

The initial noise uses an EXACT original64/128-frame draw. Independent fixed-size
tail draws preserve both prefix noise and the native global RNG state. This
avoids assuming a longer CUDA randn shape automatically preserves old prefixes.
Original source/away latent equality is verified after generation, not inferred
from the noise protocol alone. Native solver uses dynamic shifting=false.

Reuse the already-gated two-GPU VAE/output pipeline for bounded delivery buffers;
all generation, VAE, transfers, backpressure, encoding and full delivery remain
charged. This is not a new VAE optimization. Whole input noise and returned
latent arrays still grow with length, so total GPU memory is not claimed bounded.
Report actual noise storage, including fixed-draw tail padding.

First execute one96-latent two-GPU gate with exact base noise/source prefix checks.
Then30-second original-resolution strong baselines;2-minute/10-minute jobs only
after shape/memory/complete-delivery gates, on appropriately assigned H200 lanes.
