# Third-party notices

This branch contains original LongLive sparse-routing adapters and references
the following public sources. Model weights and generated media are not part of
the repository.

| Project | Public URL | Pinned commit | Repository license | Use |
| --- | --- | --- | --- | --- |
| LongLive InferHub mirror | https://github.com/Nanbuqaq/longlive-inferhub.git | `fc494740e9bf8c6bc9d0f3cbe01e68c7a2fd9fc7` | Apache-2.0 | Validated LongLive runtime and FA2 entry |
| LongLive-RAG | https://github.com/qixinhu11/LongLive-RAG.git | `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a` | Apache-2.0 | CPU history archive and latent frame retrieval reference |
| Sparse-VideoGen | https://github.com/svg-project/Sparse-VideoGen.git | `f89aedaf169ac2ae5b186bda674e53c3dc08c476` | Apache-2.0 | SVG2 parameter/formula reference |
| SVOO | https://github.com/Mutual-Luo/SVOO.git | `e4ae67b579766bcbe820bda7d34e104ff4c82d5f` | Apache-2.0 | Co-clustering and variable-block reference |
| AdaCluster | https://github.com/USTC-MLSys/Adacluster.git | `e7bed1c475a596ca6057fa7da2e5b3c37909b536` | No root license found | Formula reference only; no source copied |
| TetherMem | https://github.com/lichen1015/tethermem.git | `f9ebf718995ad162aa5b32e96a34b36a01a6c2d7` | MIT | Region/age routing and mask-contract reference; any LongLive-stack adaptation uses a distinct method id |
| SAM2 | https://github.com/facebookresearch/sam2.git | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0 | Offline oracle masks and causal first-chunk initialization; source/checkpoints remain external |

The repository-level LongLive and LongLive-RAG releases are Apache-2.0. Some
historical source headers still mention the projects' earlier
CC-BY-NC-SA-4.0 license. We do not rewrite upstream files; public submodules
remain unmodified, and local modifications are documented with source hashes.

The local `fp8-sparse-attn` fixed kernel has no verifiable license and is not
copied, published, or required. `fixed64_rect` in this project is an original
clean implementation.

TetherMem's public release uses a different LongLive-RAG checkpoint/runtime
stack.  This branch does not call a current-stack adaptation an exact
checkpoint reproduction.  Dense-reference full-video masks are offline oracle
evidence only and are excluded from causal online Pareto claims.
