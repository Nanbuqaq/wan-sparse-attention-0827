# Proxy/refresh factorial and candidate-order control

Triggered by completed `d6b20e4` development probes and `79b83a3` replays.
These bounded analyses do not change the live selector or formal holdouts.

## Frozen before the follow-up GPU evaluation

- Use the existing two prompts × layers0/19 × latent30/114, first call0 and
  clean-context call4. No additional video generation or utility tuning.
- Verify raw/roped historical KV and coordinate order are immutable between
  calls. Four equal-token routes: raw-first, raw-current, aligned-first,
  aligned-current. Aligned K is reconstructed exactly from RAW archived K at
  fixed historical RoPE positions; Q summaries use the appropriate captured
  post-RoPE Q. All teacher outputs are evaluated after routes are frozen.
- This separates prototype representation from refresh timing. Initial 8/8
  aligned-current improvements over raw-first do not by themselves prove a
  benefit from refreshing, much less from a production commit-refresh policy.
- The state pulse's `newest` candidate SET equals the baseline's six frames,
  but order differs. Its nonzero latent difference is NOT a newer-memory
  intervention. CPU layer0 replay reproduces the actual route, then confirms
  all logical edges survive sorted/reverse/rotate candidate ordering while the
  ordered union and route SHA change. GPU replay must separate FP32 and BF16
  Attention differences, and layer19 is an additional bounded check.
- Different overlap with the baseline set is another confound in whole-set
  oldest/newest/random comparisons. Follow-up causal age experiments need a
  fixed victim slot and equal replacement count, plus permutation-only control.
- Early pulse videos preserve the full common causal pixel prefix. They show
  altered ball position/cup details, not a demonstrated semantic quality gain.
  The cup is already nearly full around the intervention; this cannot establish
  continued irreversible state progression or identity/state lifecycle roles.

No legacy results are rewritten and no routing method is promoted by this audit.
