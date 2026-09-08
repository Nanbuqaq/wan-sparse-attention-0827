# Native clean-commit logs: exact KV rematerialization, bounded evidence

## What is stored

The native pipeline runs4denoisingforwards thenonezero-timestepcleancommit per
8-latentchunk. Record that actualcleaninput, exactencodedcondition and execution
directives (chunkpositions, RoPEpolicy/offset andnativepin schedule). The tested
cleaninputs were BF16 and equal the savedvideo latents. Storeencodedconditions
withdeduplication. KV samples/full witnesses are isolated validation data, NOT
part of the log or an online selectorinput.

This is model-based rematerialization, not a self-contained compressioncodec:
the same modelweights, code/backend/precision and relevant executionconfiguration
are required. It does not rerun the4stochasticdenoisingsteps or regeneratefuture
video as a teacher.

## Measured local4090 gates

Both use native5B, local32/sink8, explicit512×896gate resolution,48latentframes.

| Gate | Actual log | PositiveKV checked | Replay work | Correctness |
|---|---:|---:|---:|---|
| End-of-prefix48 | Serialized33,430,246bytes |5,284,823,040bytes, all30layers |2.2243s plus.0736s cacheallocation | EveryfullK/Vandendpoint/pinmetadata exact; step samples exact; originalnoise/latent/RGB exact |
| Pauseafter32, thencontinue16 | Prefixserialized22,287,602bytes |5,284,823,040bytes, all30layers |1.3869s plus.0735s allocation | KVactuallyreleased/rebuilt; entirefuture16latents andfullRGB exact |

The secondgate usedonly the alreadycommittedprefix andisolatedRNG. The paused
forward's cachelistaliases were cleared beforefreshallocation, preventing an
unintended secondlivecopy. No incorrectKV/original-KV fallback was permitted.

Full audits took~49/~60s because fullKVwitnessD2H, hashes andnumerical comparisons
were included. Those validationcosts are NOT the replayservice numbers above;
neither number is a deployedvideo speedup. The log/currentKV comparison is for
this finiteprefix and this32-frame cache, not a universalcompressionratio.

## Useful hypotheses and limits

- Clean commits make storedKV derivedstate in these native Densegates. Memory
  can potentially be held as a cheaplog when cold, and as KVwhen reuse justifies it.
- Log replay cost grows withprefixlength; thelog itself also grows. This is NOT
  yet bounded infinitehistory or a multi-session restorationservice. The later
  recorded-recipe experiment below does establish single cold-process restoration.
- Same-loop continuation passed, but no concurrentprimary/scratchpipeline or
  productionasynchronous checkpointservice is claimed.
- Do not transfer this proof blindly to sparseFinal: a cleanforward may consume
  a RoutePlan frozen from a noisydenoisingpass. Exactreplay could additionally
  require the actualpast routing/bias decisions andexecutionordering. This is an
  untested extension, not something already implemented.
- Native5B defaultK is alreadyabsolute-RoPE positioned. Older1.3Braw-Ksnapshot
  handling cannot be used unchanged. Everyversion/position contract remains.
- OrdinaryLLMprefix-prefill andrematerialization are priorart concepts; novelty
  of a video-specificmemoryhierarchy/co-design requires separate sourceaudit and
  real end-to-end evidence, not this primitivealone.

## Next gates

1. Matched rawKV versuslog restoration costs with boundedpinnedstaging and30
   randomizedblocked repeats; do not assume fewerbytes means lowerlatency.
2. Actualepisodeuse: rawoldKV andlog-rebuiltoldKV must have thesameadmissionSHA
   andproduce exactlythesame completevideo, while relevant/wrongepisodecontrols
   test informationvalue. The sixteen H200 full-video batch now passes this
   equivalence contract, but its global-prefix admission fails stable quality;
   see the later episode result below.
3. Only afterpositive quality/cost evidence: causal admission, checkpoint/logtail
   policies and sparse-route-decision logs. No automaticmethodpromotion yet.

Evidence:
`../../results/videos/sprint24h_20260907/longlive2_native_clean_replay48_v1/`;
`../../results/videos/sprint24h_20260907/longlive2_native_inflight_replay48_v1/`.

## Cold-process numerical recipe: identified and closed

Two fresh-process attempts failed fullKVhash checks BEFOREtiming. Cachemetadata
matched, while layer0smallnumericaldifferences amplified intodeeperKV. The native
adaLNautotuner had selected8warps; offlinecompatibilitysearch found16warps/1stage
matched every originalKVhash. This was a diagnosis with an isolatedwitness, not
an online method or proof that the oldlog was self-contained.

Newlogs now automaticallyrecord the numericaldispatch recipe, sourcehash,
Torch/Triton/CUDAversions, compute capability andinference/gradmode. Explicit
single-configTriton dispatch bypassesthe autotunecache; that path is recorded
explicitly too. The initialmissing-single-configrecording defect is preserved
as `longlive2_native_recorded_recipe48_v1` failure, fixed in v2.

`longlive2_native_recorded_recipe48_v2` passes same-processrematerialization and
fullfuturevideo equivalence. A separateprocess thenrestoresallKVusingthatrecorded
recipe, WITHOUTrecipe search orusingwitnesses tochooseparameters; fullwitness
hashes are only acceptance checks. `native_restore_benchmark_v4_recorded_recipe`
passes5warmup/30randomizedblockedrepetitions. On itsphysicalGPU1/RTX4090:

| Restorepath | Median | p95 |
|---|---:|---:|
| RawpageableCPU→GPU |.3057s|.5003s|
| Rawwithboundedpinnedstage, includingCPU pack |.5006s|.6800s|
| Cleanlog withrecordedrecipe |1.3655s|1.3703s|

The22.3MBlog ismuchsmaller thanthe5.28GBpositiveKV state, but warmrawrestoration
isfaster here. These are checkpoint-state restorationmeasurements, notvideo
speedups, notcoldNVMe/networkresults, andnotactualprocessRSSreduction (bothforms
are held forcontrolledtesting). PriorGPU0diagnostic timings are notpooled with
thisGPU1run. This also motivates includingexecutionrecipes inreplayable state.

## Sixteen H200 full-video admission/storage results

Two generated-information scenarios × seeds20260913/14 × none/raw_reveal/
raw_away/log_reveal completed at native704×1280,128latents/509pixels. Every
four-arm group has the same pre96 latent and noise. Raw/log relevant admission
has the same SHA and **bitwise identical complete latent and RGB**.

Raw archive tensor bytes2,595,225,600 versus clean log28,803,264 (90.10×).
The log demand path uploads41,386,176bytes including repeated conditions and
installs2,595,225,600bytes GPU→GPU. Archive size is not processRSS or transfer
ratio. Neither source selection nor return trigger is an autonomous selector.

Own-source visual review found partial face/red-bead retrieval in one seed each,
but flower contamination, fragments and incorrect state persist. All four
groups fail a stable overall-quality interpretation. This closes a useful
physical-storage equivalence result without promoting the admission algorithm.

Full evidence:
`../../results/videos/sprint24h_20260907/longlive2_native_episode509_h_v1/`;
`../../results/metrics/sprint24h_20260907/longlive2_native_episode509_review_v1/INTERPRETATION.md`.
