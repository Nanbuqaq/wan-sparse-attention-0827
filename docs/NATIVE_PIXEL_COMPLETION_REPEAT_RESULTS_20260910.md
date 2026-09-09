# Three paired full-run repetitions support the native completion-stage gain

Frozen runtimef4f69cc, same native509 input and two4090s. All6 executions pass
actual full latent/decoded RGB and buffer-ownership audits; no missing/failure
is excluded from the result. Three pairs were frozen before execution.

| Pair | Inline delivery | Completion-thread delivery | Latency reduction |
| --- | ---: | ---: | ---: |
| 0 | 99.418689s | 84.814404s | 14.69% |
| 1 | 96.983557s | 84.969673s | 12.39% |
| 2 | 98.950279s | 84.677274s | 14.42% |

Mode medians98.950279/84.814404s; all pairs exceed the registered10% criterion.
First-packet medians4.855152/4.939286s do not establish a first-packet gain.
Producer backpressure is6.11–7.71s inline versus~0.0002s threaded.

This is a qualified option for this workload/hardware pair, supported by earlier
real Nsight CPU-output/GPU1 overlap and exact full outputs. It adds one CPU
completion worker and43.254MB pinned output memory; no decoder arithmetic,
transfer payload, model weights or algorithmic selection changes. Model loading
is outside the timer, but pipeline construction and full delivery are inside.

Do not infer a population confidence interval from three pairs, generalize to
other hardware/motion patterns, or claim equal-resource superiority over two
independent single-GPU jobs. Give those baselines the same output optimization
before a throughput comparison. No new quality sample or SOTA ranking is created.

Facts: `results/metrics/memory_activation_20260910/native_pixel_completion_repeats509_v1/`
repeat0/1/2.json and summary/summary.json/index.html. Session35105 is closed;
the frozen six-run batch must not be submitted again.
