# LayerRecall overlap: do not rebrand known questions as novelty

Primary source read: [arXiv2608.28460v1](https://arxiv.org/html/2608.28460v1),
cached HTML SHA25660fd67abc27a8d9e2db9579e3715c4d2cdfa5b340dba6bd5d1958ae3188ca0cb.
Checked main method/setup and Appendices B.1/B.3, not just the earlier short card.

- Current-conditioned retrieval and layer-selective use are already central to LayerRecall.
- Its router is trained with CHPM;1,648,416 trainable parameters. We do not train or claim to reproduce it.
- Reported setup distinguishes32 attention-visible latent frames from80 physical-cache frames; do not inherit its speed/storage claims into our CPU-archive pipeline.
- LongLive2 policy is layers[4,9,10,12,13,15,16,17,18,26], selected via profiling plus controlled policy comparisons, not literal top-k of one mass statistic.
- Our earlier0/14/29 capture subset misses all of those layers. It cannot characterize the whole network's useful memory locations.
- Directly testing that fixed prior would be a prior-informed baseline/control, not a new layer-selection invention.

Our verified contributions so far are bounded observations: valid-source2x2 state-cue/KV interaction with retained late failures; matched-mass versus within-source readout; real pin/byte/prefix audits; and lossless fully accounted CPU-output/GPU-decode scheduling.
Whether these support a distinct robust memory mechanism remains open. Do not claim novelty or SOTA from these local controls or compare unmatched paper scores.
