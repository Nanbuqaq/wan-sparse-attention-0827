# Wave 1: native 5B resident-history interface

This development slice preserves native projections, cache writes and absolute
RoPE. It changes the Attention dispatch in memory, records the original and
derived forward SHA, and saves the derived function with each generated case.
Third-party source files and released weights are not modified.

Native mode does not install the bridge. Identity mode constructs ownership and
committed summaries but calls the exact original Attention inputs. Sparse modes
select original resident KV in one shared subset across heads; per-query/head
routes and CPU archive onload are NOT implemented by this slice.

Each clean generator call summarizes only its newly committed K/V into per-frame
Block64 means/counts (880-token frames have a 48-token tail). Summaries become
available only after that complete generator call. Only currently resident frame
summaries are retained. Current chunk, global/effective sink and actual pinned
source remain protected; the exact eligible denominator is reported per call.

The first A control uses mass times V norm. The contrast candidate subtracts the
same query's eligible-history summary mixture before computing the V norm. Query
samples are 32 deterministic actual post-RoPE queries; nonlinear scores are
averaged across queries and heads afterward. This history-only proxy is neither
full Attention probability nor exact deletion error. The GPU teacher directly
recomputes softmax on the remaining original KV. No candidate validates itself.

The D arm reuses token indices within the same chunk's denoising calls, conditional
on unchanged actual window ownership; clean commit always recomputes. It never
caches Q, noisy current KV or Attention output. All native window materialization
still occurs before dispatch. Therefore any claimed improvement must include this
preparation and remain relative to the strong unchanged native path.

The frozen JSON defines a two-case 64-latent low-resolution identity gate and ten
509-pixel development cases (toy/bead, native/mass/contrast/reuse/recent), not new
formal holdouts. The large-shape component gate uses real CUDA and an independent
FP32 teacher, but synthetic tensors and sampled query rows do not prove quality.
All lanes record failures independently and preserve their outputs. New output
roots prevent an old successful case from being regenerated accidentally.

Budgets are explicitly distinct: resident candidate logical edges are capped;
CPU raw archive is zero; full native GPU KV remains allocated; summary/index and
temporary KV costs are additional. Actual history H2D is zero in this resident
slice, which is NOT a transfer saving over native. Broad B/C/E/F exploration and
remote archive integration remain subsequent independent slices.

Evidence recovery already completed: all twenty H200 study cases were copied
with matching source/destination file hashes and fully decoded by the existing
reviewer. Visual semantic review is separate. The compact observer audit found
three original absolute-error failures whose BF16 rounding floor already exceeds
0.02; the fourth is not explained by nearest rounding. Original gates remain
failed and the second seed is not released by that audit.
