# Formal visual review protocol

Frozen before inspection of the new formal477/957 images. This is an **unblinded
assistant visual audit**, not a human preference study, exhaustive frame-by-frame
review, or statistical test of absolute quality.

Review every unique latent trajectory using all four quarter boards (16 samples
per quarter), the overview, and full-resolution boundary/detail frames where
needed. Systems with identical complete latents share one semantic review; they
are not independent quality samples. Record the panel/detail source hashes.

Use these axes independently (no automatic overall average):

- Subject identity, background coherence, action continuity and late-quarter
  visual quality: integer1–5. 1=severe failure;2=frequent obvious defects;
  3=usable but visible drift/artifacts;4=minor defects;5=stable at inspected samples.
- Category completion:0=not achieved,1=partial,2=clearly achieved in samples.
  For painting, distinguish retaining an already painted region from continuing
  to increase coverage. A stable early plateau is not continual state updating.
- State retention: `stable_sampled`, `sampled_regression`, `uncertain`, or
  `not_applicable`. Record observed reset/loop/freeze/cut evidence explicitly.
  Absence in samples is never a count of zero events over the entire video.
- Note subject, background, state and final-quarter observations in plain text.
  Dense is evaluated on the same axes; do not assume it fulfills the prompt.

Relative fidelity is separate: canonical uint8 RGB is converted to float32/255
before locked LPIPS/SSIM/PSNR, with full and quarter/late-quarter reporting.
PSNR uses the existing100 dB identical-frame cap; a high mean can reflect an
identical early prefix. Always retain late-quarter LPIPS and per-frame values.

Formal observations cannot change routes, cache budgets, backend, prompt choice,
or the independent957 seed. Failures remain in the formal result table. Speed
is reported from complete generation/VAE/artifact wall time, with model loading
and outer wall phases separately retained; evaluation time never substitutes it.
