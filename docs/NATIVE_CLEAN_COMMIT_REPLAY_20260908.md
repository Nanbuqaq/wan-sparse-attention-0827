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

## Hybrid committed checkpoints + log tails (final boundary probe)

`native_restore_benchmark_v5_hybrid`, frozen
`ce7dfa1bbabc809a3ef051ecb7fd7aef206ec186`, physicalGPU0/RTX4090. New independent
process, recorded recipe, no numerical configuration search. It creates compact
checkpoints only after the selected past clean commit (including native pin),
then restores occupied KV slots and metadata and runs ONLY remaining clean-log
records. Every path passes all original full positiveKV hashes before timing,
after warmups and after measurement. No new video is generated by this probe.

| Stored form | Tensor-state MB | Restore median s | p95 s |
|---|---:|---:|---:|
| Pure clean log |22.282|1.3932|1.3965|
| KV checkpoint8frames + remaining log |1,337.918|1.2394|1.3924|
| KV checkpoint16frames + remaining log |2,653.554|1.0765|1.3437|
| KV checkpoint24frames + remaining log |3,969.189|.8825|1.2842|
| Full raw pageable |5,284.823|.5273|1.1404|
| Full raw bounded-pinned, including pack |5,284.823|.5694|.8210|

All six paths receive5warmups and30blocked-randomized repetitions in this run.
Frame counts in this checkpoint table refer to latent frames, not decoded pixels.
Do not substitute earlierphysicalGPU1/v4times into this table. Checkpoint
construction/eviction/cold-storage IO and model loading are excluded. The
benchmark holds all forms for controlled testing: these bytes are the retained
tensor representation per path, not measured processRSS savings. Recipe and
other serialization metadata are not counted as tensorbytes.

The intermediate median tradeoff is real in this finite state; p95 improves much
less for hybrid paths. Pinned raw restoration has a slightly worse median but
better p95 than pageable in this run. Thirtysamples do not establish a production
tailSLO or a universally better scheduler. The next systems question is which
state form and checkpoint interval fit a measured revisit/deadline distribution,
including creation and background interference, not another presumed speedwin.

Structural motivation only: at native704×1280, one latentframe contains880
tokens;30layers×K/V×3072channels×BF16 is324,403,200bytes, whereas its cleanlatent
48×44×80×BF16 is337,920bytes (960× geometry expansion). Conditions, dependency
prefixes, checkpoints and model costs mean this is NOT an achieved universal
960× compression or an infinitehistory claim. The measured episode ratio is90.10×.

Plot/data: `../../results/metrics/sprint24h_20260907/native_hybrid_restore_figures_v1/`.

### Descriptive deadline counts, not a productionSLO

At the posthoc0.75s threshold, this run has rawpageable20/30 completedrestores
and boundedpinned28/30; at1.0s,26/30 versus30/30. This reinforces examining the
whole distribution, not selecting solely by itsmedian. It is oneprocess's30
repeats, not30independent sessions or a provenserviceSLO.

An explicitly UNMEASURED serialCPU-staged migration model,
`state_bytes / external_bandwidth + measured_warm_median`, crosses between raw
andlog endpoints at~6.077GB/s with these numbers. The testedbandwidthgrid's
minimum is always anendpoint; thehybrid forms do not automatically become the
fastestnetworkmigration path. Their immediateuse could instead be per-object
hardmemory caps. This is a modelinghypothesis for experimentaldesign, NOT
measurednetwork/NVMe/GPUDirect/overlap speed or anonlinecostmodel calibration.
Adding medians also doesnot reconstruct anactualcomposedlatency distribution.

Data/assumptions: `../../results/metrics/sprint24h_20260907/native_restore_decision_frontier_v1/INTERPRETATION.md`.
