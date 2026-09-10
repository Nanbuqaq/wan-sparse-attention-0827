# Native source coverage experiments

Run `scripts/run_native_causal_block_wave.py --config configs/system/native_source_coverage_wave.json --stage screen --assets INPUTS --output NEW_DIRECTORY` to inspect the frozen plan. Add `--run` only with assigned devices or inherited local physical GPU locks. The official batch entry accepts `NATIVE_BLOCK_CONFIG=configs/system/native_source_coverage_wave.json` and four or eight assigned GPUs. Actual GPU model is recorded; `NATIVE_ALLOW_H800=1` explicitly accepts homogeneous H800 as well as H200.

The six arms are native, full source, spatial mean, spatial normalized query peak, temporal groups, and a contiguous grouping comparator. The partial arms retain 1760 original tokens per head from one 7040-token source bank. This is not a quarter of the total CPU archive or global attention work. Source selection, position and first-return lifetime are shared. All arms use native in-place cache updates and shared preencoded text storage.

For the query variant, `u = p * norm(meanV)` and `score = max_query(u / sum_groups(u))`; the mean variant averages `u` over sampled queries. Both use uniform32 current Q samples and summaries of past source K/V. The temporal grouping joins one 2x4 spatial patch across eight source frames into 64-token groups. Its contiguous comparator has the same 110 groups of 64 tokens. No source mask or teacher output enters these candidates.

Use `scripts/probe_native_source_coverage.py` for the separate offline diagnostic. Its teacher recomputes attention from the remaining original K/V after deletion with fresh softmax normalization. The compact candidate formula is a proxy and is not an exact deletion error. Captured geometric Q samples differ from live uniform32 sampling.

`--native-shared-conditioning` is optional in the direct native runner. It stores each unique preencoded CPU prompt once on GPU and supplies per-block views. It requires batch-one CFG1 text-to-video, eight-frame blocks and no prefill/continuation. `--audit-shared-conditioning-inputs` checks each generator input and unique tensor integrity, recording the additional audit transfer and host cost. The default reference preparation path remains available.

Keep per-case outputs, raw route indices, summaries, logs and failures. Audit actual pre-return latents, decoded RGB, own source/away/return/late video, memory and transfer ledgers before interpreting quality. Single runs do not establish speedup or generalization.
