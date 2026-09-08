# Privileged episode admission × physical storage — frozen probe

The native scene-cut screen now forms targets, establishes a fully target-absent
final128-away-pixel interval, and returns with wrong identity/state. This is the
first useful discriminator after the failed continuous-camera-pan workload.

Four arms, all native5B/local32/sink8/absoluteRoPE/FA2:

1. `none`: unchanged native baseline, no historical CPUKV.
2. `raw_reveal`: capture the last8completed KV frames before away, then replace
   the existing global-prefix8slots once, immediately before firstreturndenoise.
3. `raw_away`: same8frame/bytebudget from the first NON-pinned awayblock. It is
   no longer resident atreturn, so this control does not merely duplicate the
   currentwindow. Fullscreen source56–63, versusreveal40–47.
4. `log_reveal`: same logicaladmission asraw_reveal, but store only pastclean
   inputs/encodedconditions/executiondirectives through the revealcommit and
   rematerialize its KV in an independent scratchcache atreturn. Restore current
   livecaches/modelsettings afterreplay and preserve RNG. No current/futurelatent
   enters the archivedepisode. No originalKV is retained bythis arm.

Important: native defaultK is ALREADY absolute-RoPE positioned. Preserve those
stored historical positions; do not apply1.3Braw-K relocation logic or doubleRoPE.
All arms retain the sameAttentioncapacity. Raw/log reveal must share the exact
admission-coordinateSHA and fullgeneratedtrajectory. Raw storage has3GiB cap,
log64MiB cap. PayloadD2H/H2D, D2DKV, peakCPUarchive anddemand wall are charged;
native tinycontrolcopies are not silently included in payload-only counters.

This is PRIVILEGED admission (episodeandreturnboundaries are experimentinputs),
not an autonomous online retrievalpolicy. A positive result would establish
memory-content value and a physicalstorage alternative, not fullco-design
promotion or a universal speedwin.

Gate:64latents/253pixels, explicit256×512resolution, starts0/8/16/48, local32.
Fourarms, seed20260904; requireallpre48latents/noise identical and raw/log full
latent/RGB identical. Lowresolution is technicalgate only. The longerawaygap
ensures both relevant andwrongmemory insert8unique nonresidentframes.

Conditional full: existing two generated-information developmentscenarios ×
seeds20260913/20260914 × fourarms=16native509executions, one frozenbatch after
allGPUbranches pass. Requirecommonpre96prefixes, raw/log fullidentity, meaningful
own-source visual/state review andwrong-memorycontrol. Stop ifrecall doesnot
reliably help. All failures/negatives and native-reference limitations remain.
