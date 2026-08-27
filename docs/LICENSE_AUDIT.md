# License and redistribution audit

Updated: 2026-08-27

| Component | Upstream / source | Revision | License status | Public repository policy |
|---|---|---|---|---|
| Sparse-VideoGen / SVG2 reference | `https://github.com/svg-project/Sparse-VideoGen` | `f89aedaf169ac2ae5b186bda674e53c3dc08c476` | Apache-2.0 | Record parameter provenance; redistribute only files carrying the upstream license and attribution. |
| SVOO | `https://github.com/Mutual-Luo/SVOO` | `e4ae67b579766bcbe820bda7d34e104ff4c82d5f` | Apache-2.0 | Vendor only the minimal required files, retain the license, and mark local modifications. |
| AdaCluster | `https://github.com/USTC-MLSys/Adacluster` | `e7bed1c475a596ca6057fa7da2e5b3c37909b536` | No repository license found | Do not vendor upstream source. The adapter is a clean-room implementation based on the paper and public interface description. |
| SCOPE | Paper `arXiv:2608.12780` | paper-derived | No official implementation found | Mark the implementation as paper-derived; do not imply official code equivalence. |
| Previous local fixed64 kernel | unlicensed local checkout | no traceable upstream revision | No redistribution license found | Never publish or vendor. Stage-2 uses `adapters/kernels_fixed64.py`, an independently written clean-room implementation. |
| SVG-EAR-derived portions present in the pinned SVOO file | SVG-EAR public integration notice | see comments in vendored source | Reported Apache-2.0 by upstream integration | Retain the original integration notice and applicable Apache license when the affected file is redistributed. |

The public publishing script must fail if it sees the unlicensed local fixed64
files, AdaCluster upstream source, model files, weights, videos, binary
extensions, caches, or internal absolute paths in the staged tree.
