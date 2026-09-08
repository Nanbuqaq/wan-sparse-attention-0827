# Streaming957 local extension — frozen before new trajectories

The477 H2002×2 established exact trajectories and mixed speed effects.
Extend ONLY the already gated complete-video streaming axis to240latents /
957pixels. This is a new length/implementation control, not a rerun of old957
formal outputs and not admission/memory-method promotion.

- Same legacyFinal route, original RoPE, exact-compact transfer, validated route
  metadata reuse, per-chunk cache and existing shared common optimizations.
- Two arms: batch/current-stream versus async-priority/current-stream VAE and
  incremental encoding. Neither enables experimental history H2D/D2H overlap.
- Development motion/state prompts, seed20260904. Both were already used for
  development; neither is a new holdout.
- GPU0 motion, GPU1 state, actualRTX4090; each arm3fresh full trajectories in
  blocked randomized order, order seeds20260911/20260912. One loaded process
  per lane,21latent untimed warmup.12new957 video executions in total.
- Unchanged, gated execution source3d00d40cb42b9a35fb7823805d91802b5904ddea at
  `/tmp/longlive-snapshot-probe.OBGzgz/checkout`. This commit contains the same
  streaming implementation as the completed477 factorial; the unrelated snapshot
  probe is not enabled in these runs.
- Require exact noise, complete ordered routes, full latents and rawRGB across
  all repetitions/arms. Report complete generation+VAE+encode/flush, first muxed
  packet, peakGPU and CPU archive independently. No client-display measurement.
- Retain negative repetitions. Do not pool these absolute times withH200/H800
  or infer a superior new algorithm from a system-only equivalence experiment.

Local launch: `../scripts/run_sprint_streaming957_lane.sh <gpu> <prompt>`.
Output root: `../../results/videos/sprint24h_20260907/streaming957_local_v1`.
The H-cluster is currently occupied by other authorized jobs; the two local
GPUs can perform this useful length extension whileLongLive2 assets download.

## Terminal result

All12executions completed with exactnoise/routes/latent/RGB. Motion paired
speedups1.1244/1.1325/1.1416 (median1.1325); state1.1552/1.1342/1.1637
(median1.1552). No negative paired repetition in this small local extension.
These are same-process blocked repeats, not independent machines.

Firstmuxedpacket moves from~333–355s to~5.3–7.1s. PeakallocatedGPU is about
13.53–13.56GB batch versus12.33–12.39GB streaming. Each trajectory still retains
65,558,937,600CPUrawKVbytes: this optimization is NOT bounded long-term history.

Steady server sink cadence is~3.81–4.12s per12pixels (p95~4.00–4.38s), far
above the.75s budget for16fps playback. Therefore earlier delivery and~13–16%
speedup are real, but real-time16fps on these4090 configurations is NOT achieved.
No client-display measurement was made. See `streaming957_local_audit_v1.json`
and `streaming957_delivery_review_v1/summary.json` in the sprint metrics root.
