# Three frozen paired full-run timing repeats after trace qualification

Reuse the gated runtime/weights/seed20260919/settled-bead509. Three pairs:
inline→thread, thread→inline, inline→thread. Both modes use the SAME qualified
strict-checkpoint constructor, two physically locked4090s, native BF16/FA2,
latent slots2, host pin cap128MiB, Torch threads2 and codec threads2. Thread mode
adds one CPU completion worker and one43.254MB output buffer. No compilation,
layout, decoder math or memory admission change.

After the equivalent Nsight run passes actual-output/slot/timeline checks,
execute six new timing runs in this ONE frozen batch. Every pair must pass
actual full latent/decoded-RGB and ownership audits; retain failures, do not
drop them for a speed summary. Report all pair differences, median/min/max and
first-packet latency. No p95 or population confidence interval from three runs.

The timer starts after model/T5 placement but includes pipeline construction,
generation, decode and completed encode/mux. This is not pure steady-state
kernel service. >=10% reduction in all three pairs supports only this workload /
hardware pair. Generalization and equal-opportunity one-GPU throughput remain
separate tasks, not consequences of this timing experiment.
