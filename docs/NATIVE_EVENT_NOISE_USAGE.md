# Match source and return noise across durations

`--duration-noise-alignment return_event` preserves the original base draw's
first96 and last32 latent noise frames for full-resolution probes. Only newly
inserted away frames use the independent fixed-size tail RNG. The64-latent
technical protocol similarly preserves first48 and last16 noise frames.
Each draw is used once within its video; matching happens between comparisons.

The generator/global RNG state after the base draw is preserved. Reports store
the original base-draw hash, actual return-noise hash, leading-prefix length,
owned noise storage and explicit alignment checks. The default remains absolute
frame alignment. Old frozen results retain their original protocol.

To run short and2-minute native/full-source controls in one four-GPU batch:

```bash
NATIVE_ALLOW_H800=1 DURATION_LATENTS=128,728 \
DURATION_GPU_PAIRS=2 DURATION_NOISE_ALIGNMENT=return_event DURATION_SEED=20261002 \
bash scripts/inferhub_native_duration_wave.sh
```

All GPUs must have the same accepted actual model. Verify identical source/
pre-return prefixes where applicable and identical return-noise hashes. This
controls return randomness; generated away history and absolute positions still
differ with duration. It does not establish natural access behavior or quality.
Each pair handles its short then long case sequentially, so short-case GPUs do
not remain reserved idle for the long case. Component gates run once per pair;
model cases run in independent processes and failures are recorded individually.
