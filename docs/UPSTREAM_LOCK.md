# Upstream and provenance lock

- Sparse-VideoGen: `svg-project/Sparse-VideoGen@f89aedaf169ac2ae5b186bda674e53c3dc08c476`, Apache-2.0.
- SVOO: `Mutual-Luo/SVOO@e4ae67b579766bcbe820bda7d34e104ff4c82d5f`, Apache-2.0.
- AdaCluster reference: `USTC-MLSys/Adacluster@e7bed1c475a596ca6057fa7da2e5b3c37909b536`; no repository license was found, so no upstream source is redistributed.
- SCOPE: paper-derived implementation for arXiv:2608.12780; no official code is claimed.
- SVG2 Wan 14B/720p reference parameters are Q=300, K=1000, Top-p=0.9, min-K=0.1, init=50, and step=2. They are calibration candidates, not Wan1.3B defaults.
- Stage-2 fixed64 is the clean-room source in `adapters/kernels_fixed64.py`; the previous local fixed64 checkout has no redistribution permission and is neither imported nor published.
- Vendored SVOO files retain the upstream license and modification/integration notices. Exact task manifests record the specific files and hashes used by each run.

Current public-source hashes:

- clean-room fixed64: `167b161ad6d39589759ae63985a0421a5655071d4a0b53598754f469416c2b3b`;
- clean-room CSR varlen: `9c0e21602f7570eb3c4f40a8a0585c8d2c304ef95f2a416bef42f05da6312eea`;
- vendored SVOO co-clustering: `8b2d1e52b7151b4d763f3ac93c7759df21e03b2f4785457a56fa2252ef6af2fb`;
- vendored SVOO permutation: `ad4d16c114fd68fab15b9f2206f47d21db12674a8ef6b15e764ba7cedacd6ce2`.

See `docs/LICENSE_AUDIT.md` for redistribution policy.
