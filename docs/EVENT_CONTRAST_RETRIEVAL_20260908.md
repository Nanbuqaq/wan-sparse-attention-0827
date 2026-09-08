# Causal instruction-event retrieval candidate

The paired anchor diagnostic supports timely retrieval in two development seeds, but uses privileged episode intervals. This candidate removes those supplied intervals: it detects changes of the current immutable condition object, keeps only summaries of observed conditions and their completed time intervals, and selects the last six **available committed frames** of a past episode.

Use the existing FP32 UMT5 condition embedding, mask zero padding, mean-pool and normalize. At event i, for past episode e<i:

\[
s_e=\cos(t_i,t_e)-\cos(t_{i-1},t_e).
\]

Only the current condition and past episode keys enter this score. Future preencoded condition objects, workload role names, future frames, Dense output and manually specified anchor ranges are not passed to the selector. After at least two episodes are closed, select the highest score and, if six committed frames are available, replace coarse retrieval only during the first new chunk. Other calls use ordinary visual RAG. Insufficient history is an explicit ordinary-RAG fallback, not a Dense fallback.

This is a heuristic. On the two known input schedules, plain cosine selects the away episode on return; contrast selects the earlier reveal/changed-state episode. That text-only fact is not video-quality evidence. The candidate may select inappropriate old memories for genuinely new instructions; no confidence/generalization claim is made. The episode-key table and original CPU KV archive are not yet bounded.

## Frozen development protocol

First a real39-latent gate: scheduled Dense, cosine-event retrieval, contrast-event retrieval; no pre-intervention prefix may change. Then, conditional on the gate, the10-case477 design in `event_retrieval_negative_controls.json`:

- Original toy schedule, two seeds, Dense/cosine/contrast:6 cases.
- New duck instruction, seed20260909, Dense/contrast:2 cases.
- Explicit absence instruction, seed20260910, Dense/contrast:2 cases.

The selector receives no privileged `role` fields. All current/past-condition summary costs and actual metadata transfers are recorded. Frame count stays equal to the existing six-frame RAG budget. Current text is a declared causal input; this is an extension of the coarse-retrieval context, not a hidden teacher entry in the fine-grained utility API.

Inspect identity, extra-object injection, instruction completion and late-quarter behavior. Reject generic “more old memory is better” reasoning: persistent privileged anchoring already produced a blue-body distortion in one seed. Do not promote this candidate from its text similarity matrix alone.
