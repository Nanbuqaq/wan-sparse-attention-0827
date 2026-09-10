# Explicit homogeneous Hopper batch policy

The H-class cluster can contain both H200 and H800. The default native batch
policy still requires H200. Set `NATIVE_ALLOW_H800=1` in the submission command
to explicitly accept either H200 or H800. This is recorded in `hardware.json`;
every GPU assigned to the batch must report the same actual model.

Source and duration launchers propagate this policy with `--allow-h800`.
All lanes also execute the existing real CUDA/FA2 component gate before model
work. Numeric tolerances are unchanged. Hardware/topology and numeric failures
are reported separately, and failed lanes cannot invalidate successful ones.

Example command prefixes:

```bash
NATIVE_ALLOW_H800=1 bash scripts/inferhub_native_causal_block_wave.sh
NATIVE_ALLOW_H800=1 DURATION_LATENTS=728 DURATION_SEED=20261002 \
  bash scripts/inferhub_native_duration_wave.sh
```

H800 results must be labeled H800. Compare methods and costs within the actual
GPU model; never pool H800 timings into an H200 speedup. Algorithm implementation,
prompts, seed and budget configuration are unchanged by the hardware opt-in.
Use fresh output directories for recovery and retain prior failure artifacts.
