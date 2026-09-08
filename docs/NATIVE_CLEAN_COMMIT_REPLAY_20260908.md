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
  yet bounded infinitehistory or a cross-process/multi-session restorationservice.
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
   test informationvalue. Shortfour-arm gate already passes; longbatch pending.
3. Only afterpositive quality/cost evidence: causal admission, checkpoint/logtail
   policies and sparse-route-decision logs. No automaticmethodpromotion yet.

Evidence:
`../../results/videos/sprint24h_20260907/longlive2_native_clean_replay48_v1/`;
`../../results/videos/sprint24h_20260907/longlive2_native_inflight_replay48_v1/`.
