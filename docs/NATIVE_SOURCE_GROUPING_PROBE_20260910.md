# B-line first grouping discrimination

Use the existing nine real native related_recent QKV records: layers0/14/29,
phases0/3/clean4, query96. The second8-frame KV segment is the recalled source;
the other24 frames remain unchanged. No new video or online teacher is involved.

Compare flat within-frame Block64, compact spatial8x8 groups, and a flat partition
with exactly the same group counts/sizes as spatial8x8. This distinguishes spatial
membership from the otherwise confounding difference in number of groups/tails.
Each partition is scored with the registered mass/value and contrast/value proxy.
Add a deterministic token-random control. All variants select exactly1760 source
tokens; trim the final ranked group only for this offline controlled comparison.
Do not call that the live whole-block policy or infer physical layout savings.

Teacher directly recomputes FP32 Attention on the actual remaining original KV,
including fresh normalization and all protected context. Report output error and
retained source mass; preserve original BF16/FP32 numeric gates per capture row.
The capture contains only32 geometric query sites, unlike live uniform32 sampling.
It tests a grouping hypothesis, not full-Q accuracy, closed-loop quality or novelty.

Mechanistic question: the native880-token frame makes flat64 blocks long spatial
strips. Can compact grouping better represent relevant source information at the
same raw-token budget, without assuming a semantic mask? A positive result only
earns an online source-admission test with archive-time grouping, actual packing
and maintenance costs, and full-source/native controls.
