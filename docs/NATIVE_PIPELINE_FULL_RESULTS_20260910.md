# Native full-resolution pipeline gate after research resumed

Frozen runtime: 444ebcd3c9752404cc41966d8b3f86cd908cfb3a, snapshot
`/tmp/longlive-native-pipeline-full.bdwXxP/checkout`. Both physical 4090s were locked.
One native 704x1280 / 128-latent / 509-pixel trajectory, seed20260919,
settled_bead_revisit, native BF16/FA2, CFG1 positive-only allocation, local32.
Reference: existing settled_state_screen509_v1/lane0/settled_bead_revisit.

Both serial and overlap passed the independent audit: actual complete latent
and decoded RGB exactly match the reference, streamed latent checksum matches
the returned latent, prompt/pin identity and all frame boundaries agree.
No new independent quality sample or quality improvement is claimed.

| Same two-GPU resource control | Serial | Bounded overlap |
| --- | ---: | ---: |
| Warm pipeline delivery, seconds | 172.734965 | 98.865918 |
| Generation host span including backpressure, seconds | 172.732804 | 87.063041 |
| First muxed packet from generation start, seconds | 4.926939 | 4.837661 |
| GPU0 peak allocated, bytes | 24378089472 | 24380792832 |
| GPU1 peak allocated, bytes | 12133365760 | 12133365760 |

Pinned input5,406,720 bytes and pixel staging43,253,760 bytes, below128MiB;
latent D2H/H2D each43,253,760 bytes; pixels D2H5,504,040,960 bytes.

This single pair is a correctness gate, NOT repeated speedup evidence. Load
times differ substantially (241.13 vs147.31s), so no cold-latency ratio is used.
VAE placement/model/T5 initialization are recorded separately; complete_s starts
at generation. Two provisioned GPUs also cost resources: latency is not aggregate
throughput, energy efficiency, or equal GPU-second advantage over one-GPU service.

The CPU host trace is not GPU-overlap proof. The next bounded diagnostic is
one equivalent native overlap run under Nsight, with both device kernel/copy
tracks, complete payload checks and a dual-device Perfetto export. Repeated
timing and stronger resource baselines follow only if that trace passes.

Facts: `results/metrics/memory_activation_20260909/native_pipeline_full509_audit_v1.json`
and `results/videos/memory_activation_20260909/native_pipeline_full509_v1/`.
