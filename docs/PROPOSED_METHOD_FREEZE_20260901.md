# Proposed-method freeze audit

The proposed LongLive routes were frozen before formal sparse prompts were
used. Exact output residuals were restricted to the offline QKV teacher. The
online routes use only GPU Q summaries and CPU Block64 K/V mean prototypes
before transferring and executing selected original K/V.

## Isolated QKV selection

- Analysis commit: `3942834f67028308d07a5f4502f0b3816f46c13e`
- Captures: 12, covering layers 0/9/19/29 and early/middle/late calls.
- Candidate grid: two base/local splits, Q-summary blocks 64/128/256, three
  V weights and three final transfer multipliers.
- Selected coverage: 70/15/15 with Q block 64.
- Selected V-aware: 70/15/15 with Q block 64 and V weight 1.0.
- Selected final: the same V-aware proxy with transfer multiplier 1.0.

## Two-prompt 477-frame gate

All eight same-commit cases reached terminal states with 477 decoded frames,
finite latents, grouped FA2 execution and zero fallback, failed or NaN calls.

| Method | Motion transfer | State transfer | Motion E2E | State E2E | Manual result |
| --- | ---: | ---: | ---: | ---: | --- |
| RAG Dense | 100.00% | 100.00% | 7149.2 s | 7263.3 s | pass |
| Coverage | 90.75% | 87.96% | 5972.4 s | 6093.7 s | motion negative, state pass |
| Online V-aware | 90.39% | 86.73% | 5411.7 s | 5840.7 s | pass |
| Transfer-aware final | 25.00% | 25.00% | 1949.9 s | 1750.0 s | pass |

Coverage-motion is retained as a formal negative ablation: robot identity
changes, the ball duplicates and two late flicker diagnostics are present.
The final method preserves monotonic state without reset in the state prompt;
its motion prompt retains the main robot and ball but has moderate design drift
and one late extra red object. This is recorded as light late degradation, not
hidden or promoted to Dense quality.

The final method was approximately 3.67x faster than matched RAG Dense on the
motion case and 4.15x faster on the state case in the dual-4090 calibration,
while enforcing 25% history pair and transfer density.
