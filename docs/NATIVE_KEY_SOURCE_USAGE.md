# Archive-time key grouping

Source-block grouping accepts `key_frame`, `key_bank` and `flat_key_matched`.
These paths build source groups from closed-scene GPU keys before recall
rebinding; no query or teacher is used for construction. Original raw selected
KV still goes through the existing native source install/FA2 path.

The first slice holds normalization=source_only and refresh=first_only.
Use a partial mass/value or contrast/value policy. Native22x40 token grids
produce128 groups of55 tokens; the8x16 technical grid produces16 groups of64.
The flat control matches these counts and sizes without content clustering.

CPU partition indices/counts and mean tensors are included in the scene archive
budget. Construction and lookup sub-scopes are nested in the existing group
preparation and scoring scopes, respectively. Their times must not be added
twice. GPU grouping reads and temporary allocations are real work to account for.

`source_key_partitions.pt` records all partitions in retained banks at final
export; `causal_block_routes.pt` records the actual selected original coordinates.
Evicted banks are not retained solely for diagnostics. This is a fixed-cap
balanced cosine candidate, not the complete ClusterVLM maintenance system.
