# TetherMem oracle and causal boundary

## Pinned source

The official source was inspected at
`lichen1015/tethermem@f9ebf718995ad162aa5b32e96a34b36a01a6c2d7` (MIT).
Its documented canonical pipeline is:

```text
matched LongLive-RAG reference video
  -> first-frame subject selection
  -> full-video SAM2 propagation
  -> TetherMem generation with the same prompt and seed
```

The released attention patch reads the subject mask at the current generated
frame and reads the masks of retrieved historical frames. Because those masks
come from a previously completed reference video, this is a valid offline
mechanism teacher but not an online streaming method.

## Official routing semantics retained

- Subject and background queries are split without materializing a dense
  `[Q,K]` bias tensor.
- Subject queries keep unit weight on subject memory and use a solved context
  weight on background memory.
- Background queries de-emphasize subject memory and apply recency decay to
  background memory.
- The context weight is clamped so the spatial average remains at the target
  budget when possible.
- The original path first transfers its retrieved memory and then changes
  Attention weighting. It is not evidence of online H2D byte savings.

This source-backed mode is named `tethermem_oracle_mask_teacher`. Its VAE,
SAM2, mask propagation, synchronization, and reference-video generation cost
must be reported separately. It cannot enter the online speed Pareto.

## Causal runtime contract

`causal_subject_router` uses only masks from completed latent chunks. Decoded
mask sequences are aligned explicitly from `4*T-3` pixel frames to latent
anchors `0,4,8,...`; latent and pixel frame counts are never used
interchangeably. At the end of a completed chunk, masks and their measured VAE,
SAM2, and synchronization service time are committed contiguously.

Before the next chunk, historical Block64 identity probabilities are read from
the committed masks. Current Q summaries interact with past subject/background
K prototypes to produce soft query roles. Any history block whose frame id is
not strictly earlier than the committed frontier is rejected. Role state is
updated only after the current chunk completes.

The causal path remains a candidate until it passes:

1. oracle-versus-causal role agreement and drift audit;
2. full external-model cost accounting;
3. same-budget output/video quality gates;
4. no use of manual boxes in formal online results.

If these gates fail, only the offline oracle upper bound and the negative causal
result are retained.

## Source-level temporal discrepancy found during independent audit

In the pinned public pipeline, `scripts/extract_sam2_mask.py:219` emits one
`[30,52]` mask per **decoded pixel frame**. `load_patch_mask()` only validates
shape/binarizes and does not resample time. `tethermem/routing.py:618` instead
addresses the query mask with `current_start // frame_seqlen`, a **latent**
index, and repeats this mask across the new latent chunk. Historical masks use
pool indices directly (`build_memory_subject_mask`, line 49), without the pixel
4x mapping or the sink offset. The automatic pipeline passes the mask directly.

Therefore a source-compatible oracle replay and a time-aligned mechanism
teacher are distinct variants. Do not silently call the existing aligned
adapter a bitwise reproduction of the public mask-addressing behavior.
Neither variant has completed end-to-end Tether video validation here.

Official mask tests previously passed 5/5; routing tests now pass 4/4 in a
CUDA-visible process. Their numerical test tensors are CPU tensors: this is
not a full GPU routing-kernel or video gate.

## SAM2 prefix feasibility probe

Exact source is `2b90b9f5ceec907a1c18123530e92e794ad901a4`; Hiera-L checkpoint
SHA-256 is `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`.
The checkpoint loads with 900 finite model tensors. Missing Hydra/iopath/
portalocker were installed only under this project's `.runtime/sam2-python`.

Using only image 8 of a completed nine-image prefix, an unchanged central-
foreground rule produced no eligible state mask with an 8x8 point grid. A
16x16 grid produced plausible cup and toy masks on the two development prompts;
assistant visual inspection is not semantic ground truth. Mask inference was
0.83 / 0.92 s, with 13.73 / 2.54 s separate model loads (warm-cache differences).
This is an offline prefix/API-cost probe, not propagation, VAE accounting,
role-drift validation, or an integrated online method. Connected-component
postprocessing was explicitly disabled; no manual ROI was used.
