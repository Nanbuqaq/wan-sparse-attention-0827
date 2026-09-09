# Independent state screen: chest qualified, envelope not qualified

All eight Dense videos technically passed, and all four actual first48-latent /
first189-decoded-RGB paired-prefix checks passed. The source and paired controls
were fixed before generation; no new memory method has run on them yet.

Chest sources both visibly achieve open lid + blue cloth, with no source hand
in the sampled final-eight interval. Late-away boards genuinely omit the chest.
Both returns show a closed chest; seed25 retains a large foreground daisy.
However, the visible controls also lose the open state at the later cut after
retaining blue cloth through the middle interval. Seed26's tail has at most a
small blue sliver. Thus this is not a pure long-absence memory-loss task.

Envelope sources show red-card/open-envelope content but do not satisfy the
registered hand-free settled-source requirement on both seeds. Seed25 retains
hands throughout the source board; seed26 also has hand/tear ambiguity. Both
revisit and visible tails close the envelope. Do not promote this category or
retune prompts/seeds inside the completed screen.

This is descriptive assistant inspection, not blind scoring. Source/late-away/
return and visible-control interval boards were inspected; no claim of watching
every frame in continuous playback. All technical and semantic limitations stay.

Only chest is eligible for the next frozen causal-memory verification. Keep the
existing descriptor selector parameters unchanged. Separate selecting the past
state from the key-position policy, and retain the visible-cut negative as a
boundary on any long-range-forgetting narrative. Do not promote a new three-role
router from the earlier distance-matched color-proxy experiment.

Facts: `results/metrics/memory_activation_20260910/object_state_screen509_review_v1/`
technical_audit.json, semantic_review.json and the interval boards.
