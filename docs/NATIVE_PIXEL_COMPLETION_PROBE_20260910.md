# Bounded CPU pixel completion behind native two-GPU decode

Existing full509 inline-output overlap trace records128groups and15.612s summed
CPU-pixels-ready→sink-finished spans. Source shows VAE's worker calls the sink
synchronously after each D2H readiness wait. This includes normalization, hashing,
encoding and mux; it is not15.6s of codec-only CPU compute or guaranteed E2E savings.

Add a bounded single-consumer pixel worker, two owned pinned output buffers,
and explicit backpressure/failure propagation. Keep every decoder call, float
conversion, D2H readiness wait and sink operation unchanged. Pixel buffers cannot
be recycled until the sink returns. Input pin5.407MB plus output pin86.508MB
fits128MiB. Keep old inline mode as default and compare at the same two-GPU budget.

First short, then full raw-latent/RGB equality; only then paired latency and a
matching Nsight timeline. This targets CPU output versus GPU decode overlap,
not same-device H2D/kernel overlap or lower transfer bytes. No approximate VAE
layout is used. Any one-GPU throughput claim requires giving its decoder the
same completion optimization; do not compare against an unoptimized baseline.
