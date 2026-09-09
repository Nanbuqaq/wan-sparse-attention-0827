# Real trace confirms output/decode overlap; kernel work stays essentially the same

One equivalent native509 thread-mode capture completed with actual full latent /
decoded-RGB and buffer-ownership audits. Official Perfetto parsing also passes.
Frozen runtime files match the successful8f49008 gates; no decoder math changes.

| CUPTI/NVTX diagnostic | Earlier inline | Completion thread |
| --- | ---: | ---: |
| Full measured window | 98.358618s | 84.865081s |
| GPU0 kernel service | 74.693585s | 74.818203s |
| GPU1 kernel service | 77.208995s | 77.241579s |
| Cross-device kernel overlap | 58.843661s | 69.037886s |
| CPU output-range union | 14.810258s | 14.474024s |
| CPU output overlapping GPU1 kernels | 0s | 13.337778s |

About92.15% of the new CPU output-range union overlaps actual GPU1 kernels.
This is not a claim that92% of E2E time is saved. Kernel service is effectively
unchanged in these two diagnostics; critical-path scheduling changes. Pixel D2H
remains5,504,040,960bytes, not a PCIe-byte reduction. The extra43.254MB pinned
output pool and one CPU completion worker are retained in resource accounting.

The older and newer diagnostic runs are not timing repetitions. Their role is
to demonstrate the mechanism and exact output, not provide statistical latency
confidence or one-GPU throughput superiority. A frozen three-pair unprofiled
timing stage follows; preserve first-packet and initialization boundaries.

New facts: `results/metrics/memory_activation_20260910/native_pixel_completion_profile509_v1/`
actual_output_audit.json, audit/audit.json, audit/native_two_gpu.perfetto.json.gz,
and parser_validation/validation.json. Tool session13346 is closed.
