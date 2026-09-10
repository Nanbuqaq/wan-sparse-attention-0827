# Native duration runner

`scripts/run_native_duration_wave.py` runs matched native and full-source scene
memory cases. Each case uses two assigned GPUs: generator and streamed VAE/output.
One scenario requires four GPUs; `--scenario both` requires eight. Devices come
from the platform's `CUDA_VISIBLE_DEVICES`, and each lane verifies its GPU type.

Example entry after official preparation:

```bash
DURATION_LATENTS=728 DURATION_SEED=20261002 \
DURATION_SCENARIO=generated_patchwork_toy_cut_revisit \
bash scripts/inferhub_native_duration_wave.sh
```

Use the same script with `--prepare-only` for CPU asset/source verification.
Preparation creates only the output marker; inference output roots must be new.
Failures are recorded per lane, and completed artifacts are retained.

Supported full-resolution lengths:184,728,3608 latent frames, producing733,2909,
14429 video frames at24fps. `run_longlive2_native_reference.py` also supports
64/96-latent technical gates with `--gate --episode-gate-layout`.

The duration option extends the scripted away interval and shifts the return.
Native local32/global8/pin8, chunk8 and three scene closures stay fixed. It does
not enlarge candidate K or simulate archive growth by duplicating old KV.
The complete video uses newly generated history. Partial-source methods are
excluded from this duration runner's baseline protocol.

Noise generation preserves the original128-frame draw and the native global RNG,
then uses independent fixed-size tail draws. At the same duration, both methods
share the entire noise tensor. Across durations, absolute-frame prefixes match;
the shifted return uses different absolute-frame noise. Verify actual latent and
pixel prefixes, rather than inferring equality from the random seed alone.

Reports include delivery timing, GPU allocator peaks, pinned pools, transfer
payloads and retained noise storage. Whole noise/output latent arrays still grow
with duration. Host scopes and GPU stream intervals are not additive; hardware
overlap requires a separate trace. VAE and encoding remain charged.

The native source selector also accepts `--causal-block-normalization joint_context`
for mass/value or contrast/value policies outside the duration baseline protocol.
It uses compact means of kept GPU context plus archived source means, and charges
the extra summary work. The nested kept-summary preparation interval is included
in the total scoring scope and must not be counted twice. No teacher output is
an online input; original selected KV is executed by the native FA2 backend.
