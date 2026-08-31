# Dense-only formal prompt freeze (2026-08-31)

The Dense screen used one shared code commit, eight candidate prompts, Native
Dense and RAG Dense, and seeds 20260826/20260827. All 32 videos passed the
technical audit with 477 decoded frames and zero failed/fallback/NaN calls.
Sparse results were not read during selection.

Each video was scored from 0 to 2 for category completion, subject consistency,
background consistency, continuous motion, and freeze/flicker/cut. For each
prompt and seed the lower Native/RAG total was used; prompts were ranked by the
two-seed mean, worst seed and prompt id.

| Category | Frozen prompt id | Two-seed score | Main rejected alternative |
| --- | --- | ---: | --- |
| Identity/background | `identity_fluffy_creature` | 10.0 | Astronaut remained stable but did not clearly inspect/touch plants |
| Irreversible state | `state_water_pour` | 9.0 | Candle showed almost no shortening or wax accumulation |
| Human action | `human_chef_plating` | 10.0 | Glass geometry changed abruptly and second-seed progression weakened |
| Fast motion | `fast_gymnast_ribbon` | 10.0 | RAG fox motion slowed strongly in the last quarter |

Artifact SHA-256:

- expanded 32-case diagnostics manifest: `9cbbf2810703cdd125f0df4b45d63114220da7ffc03745a73edd9a564f63e901`
- human score decisions: `c3adae360c8b6124ef0184afb6c70f570c9c6673341b34a440a652bb531d625a`
- scored review table: `8bb58a4ba0c5683de4409d3cd389b69a2303911f46cdc6c6c315812cd015e7dc`
- public frozen prompt manifest: `aef993753e9d3be5936427c781fac2569294bb24dd782c9d172c41034facf86a`
