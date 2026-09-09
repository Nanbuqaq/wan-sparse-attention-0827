# Returned state is quickly represented by newly generated KV

CPU-only source-derived lineage for the completed hybrid, native absolute-RoPE,
local32/global8/shot8, cuts16/48/96. Both seeds' actual pin event positions and
terminal pinned slots match the derived rolling rule. This is not an actual
full-KV provenance capture or an Attention-probability measurement.

| Query latent start | Direct original source40–47 slots | Other retained context |
| --- | ---: | --- |
| 96 | 8 | initial0–7, away88–95, current96–103 |
| 104 | 0 | initial0–7, away88–95, return96–111 |
| 112 | 0 | initial0–7, return96–119 |
| 120 | 0 | initial0–7, return96–103 and112–127 |

After the first returned clean chunk, native pinning chooses newly generated
96–103. On the next rolling event the original recalled source is no longer
directly present. Its information may survive in the new return KV; absence
of the original slots does NOT mean complete forgetting.

Both seeds undergo this same change, but only one closes late. Therefore it is
not a sufficient explanation of the quality difference. A bounded follow-up
could keep the original source as pin and test delayed divergence, but that
changes retained content (including which away/return slots are discarded),
not a pure layout optimization. The earlier bead source-repeat/pin results are
already complete and must not be rerun or treated as this new controlled test.

Facts: `results/metrics/memory_activation_20260910/chest_memory_lineage_v1/lineage.json`.
