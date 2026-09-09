# Is better access only more source attention mass?

CPU-only counterfactual on the already-verified native bead capture, all nine
phase/layer records. Keep original Q/V and non-source K fixed; rotate source
temporal K by the previously qualified delta64. No new video or online method.

For each sampled head/query, match the rephased source partition function using
an additive group bias `logZ_new_source - logZ_old_source`. This preserves the
original within-source attention distribution while matching total source mass
and all other group masses. Compare its output to the truly rephased output.
Also retain fixed gains2/4 and within-source distribution total variation.

The fitted per-query bias reads complete teacher K: it is an offline diagnostic,
not a permissible selector or frozen online score. Residual divided by total
rephase output change can exceed1 due to cancellation; it is not a fraction of
causally explained quality. Every original FP32/native gate must pass; only
first-denoise L0 can match the earlier actual rephased trajectory's inputs.
Later rows stay fixed-original-Q counterfactuals. Clean commit is not a fifth
denoise, and geometric query samples do not establish semantic/whole-video gains.
