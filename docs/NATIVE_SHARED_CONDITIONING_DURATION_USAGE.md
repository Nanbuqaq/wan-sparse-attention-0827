# Long-video common text preparation comparison

The native duration runner accepts `--methods native native_shared`. Both arms use the same original bounded native attention, noise, prompts, in-place KV updates and two-GPU video delivery. `native_shared` only enables unique preencoded text storage. The default `native scene_full` experiment remains available.

For the official entry use `DURATION_METHODS=native,native_shared`, `DURATION_GPU_PAIRS=2`, `DURATION_LATENTS=3608`, `DURATION_NOISE_ALIGNMENT=return_event`, and an explicit seed. Four assigned GPUs run two independent cases. Stage CPU preparation separately; preserve failed lanes and never retry successful outputs within the frozen batch.

Compare full actual latents and decoded RGB, condition storage/H2D ledgers, generator peak allocation, VAE peak allocation, complete delivery and first packet times. Charge both GPUs. A single pair validates output/storage behavior; it does not establish a repeated speedup. Hardware must be homogeneous and reported as its actual model. This comparison does not change the online memory algorithm or establish natural archive/access growth.
