# All-layer offline role statistics without dumping complete K/V

Prior0/14/29 diagnostics do not cover LayerRecall's memory-sensitive allowlist.
Run an observer-only probe on the two completed natural-hybrid trajectories:
query starts96/120, denoising phases0/3 and clean4, all30layers. Each head uses
32fixed geometric queries. First query labels are initial/source/away/current;
last query labels are initial/first-return-anchor/recent-return/current, based
on verified pin metadata and the existing lineage model.

Call native Attention unchanged. Then compute per-head FP32 group logZ and
conditional value outputs, with TF32 temporarily disabled and restored before
return. One head/group at a time bounds scratch; persist only compact statistics
and sampled native outputs under512MiB, not full Q/K/V. These support sampled
group deletion/reweight analysis only, not arbitrary new token or RoPE replay.

No probe tensors enter routing, no training, no new quality sample or valid
method timing. Full generated latent/decoded RGB must match the old hybrid.
Every sampled FP32/native comparison has its own numerical gate; failed rows
are retained, not used to choose a favorable layer policy. Do not freeze an
allowlist from one statistic or claim novelty over prior layer-selective work.
