# Recent-context allocation diagnostic

The locked profile configuration is local cache12latent, sink1, history6, chunk3, recent-exclude5. The upstream RAG code and adapter reserve six dense attention frame slots for coarse history regardless of how much raw history a sparse method eventually uses.

At current latent78, the first forward consumes exact sink0 and local frames76–80 (two completed frames plus the current three). Coarse retrieval is decided before that forward's eviction and can select only frames1–61. Frames62–75 are guaranteed absent from both paths: five excluded CPU frames62–66, newly evicted67–69, and already-resident but excluded70–75. Potentially eligible frames are not all actually selected.

Evidence: `results/metrics/sprint24h_20260907/recent_context_geometry_v1.json`, source configuration SHA, and the unchanged upstream local-budget expression. The generic `recent_exact_frames=3` sparse-config field must not be described as three completed previous frames in this runtime.

This is an intentional allocation/eligibility policy, not a demonstrated correctness bug. Sparse history can free attention work without automatically reallocating it to recent resident information. The hypothesis is that explicit larger local windows may protect changing state; it needs a matched-method video test and must also be offered to baseline Final.

`exact_local_window_frames` is therefore an optional **method parameter**, including the current chunk. It changes logical exact edges and enters case identity; it is not a TransferPlan-only optimization. The old default remains unchanged. A window8 experiment consumes up to sink1 + local8, including five completed previous frames, instead of sink1 + local5. It does not onload additional KV, but adds GPU attention work. No quality or speed improvement is claimed yet.
