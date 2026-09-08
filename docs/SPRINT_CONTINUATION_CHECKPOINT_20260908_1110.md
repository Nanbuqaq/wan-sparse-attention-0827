# Live checkpoint — 2026-09-08 ~11:10 UTC

Continue the active24hgoal through14:21UTC or userstop (~3h11m left). No training,
no subagents, old44/102/formaloutputs read-only. Do NOT resubmit or kill livejobs.
This turn made substantialprogress: exact957pipeline extension, stronger native
baselines, clean scene-cut memory failures, exactclean-log rematerialization and
a gated episodeadmission/storage experiment. No universalonlinewinner yet.

## Live work and Git

- Mainrepo `/home/zhouhe08/MyProjects/0904-longlive-system/publish_repo`, branch
  `longlive-system`. Latestlocal2b2c537 (fullSHA viaGit); lastpushed
  217c39de689634831162eb9333c23ce1f71559b5. Preserve preexisting untracked
  `scripts/audit_retrieval_divergence.py` and old0400/0500checkpointdocs.
- ONE queued/runningInferHub batch, submittedonce:
  `zhouhe08__longlive2_episode_admission_storage_Iter0__217c39de6896`.
  Root `/kaimm-distill/zhouhe08/longlive-system/outputs/sprint24h_native_episode_217c39d`.
 4GPUs,16native509videos:2scenarios×2seeds×none/raw_reveal/raw_away/log_reveal.
  Preppassed; Hpoolwasbusy and jobreturned toqueue asprepared. At10:49noarms had
  begun. Rechecklive; do not duplicate. Eachlaneusesdistinctscenario/seed and
  alternatesarmorder; failedarmdoesnotstopothers.
- LocalGPU0 runs **native restore benchmarkv2** / diagnostic, exec17544,
  output `results/metrics/sprint24h_20260907/native_restore_benchmark_v2`, log
  adjacent`.log`. It first comparesallKVhashes, then onlyifpass measures30repeats.
- LocalGPU1 isfree asof11:07; verify beforelaunch. Usephysical locks andescalation.
- Benchmarkv1failed BEFOREtiming withKVhashmismatch andisstored unchanged.
  v2runs onthe samephysicalGPU0 asoriginalinflightproof andsaves stepwise/fullhash
  diagnostics. It mayfailagain; inspectexactfirstlayer/step/errorsbeforechanging
  anything. Do NOT downgrade the strictgate to gettimingnumbers.

## Critical new positive: native clean-commit rematerialization

Code `adapters/longlive_sparse/native_commit_replay.py`, flags on
`scripts/run_longlive2_native_reference.py`.

1. `longlive2_native_clean_replay48_v1`: native5B lowres512×896,48latents/189pixels,
   local32/sink8,6cleanforwards. Serializedlog33,430,246bytes reconstructs
   positiveKV5,284,823,040bytes. All30fullK/Vandmetadata exact; allstepsamples exact;
   cleaninputs equalreturnedBF16latents. Replay2.2243s plusallocation.0736s;
   fullaudit~48.87s includes5GBwitnessD2H/hash/metricoverhead. Originalnoise,
   latentandRGBmatch unobservedgate. Source1bc1e12cbd2de2fd279a7dc4a0de7f8ba1f79fab.
2. `longlive2_native_inflight_replay48_v1`: pausesafter32latents, releasesKV,
   rebuildsonlyfrompastprefixlog, thencontinues16futurelatents. Fullvideoremains
   exact. Prefixfile22,287,602bytes vspositiveKV5.284GB; replay1.3869s+allocation
   .0735s. Fullaudit59.6s is NOTthe servicecost. All30original/replayedK/V SHA
   pairs were explicitlychecked andequal (notonly`torch.equal`). Source
   631cc80d9cd55dca9259a5d1306c4faf490ca24f.

`clean_commit_prefix_log.pt` excludesKVwitnesses; `offline_sample_witness.pt` is
separate. Logrecords actualBF16cleaninputs, exactencodedconditions, positions and
RoPEsettings; geometryis included. Model/code/backend versions remainrequired.
This is NOT astandalonecodec, notyetbounded infinitehistory, notyetcross-process
restore, andnot a demonstratedvideo speedup. Newprocess benchmarkv1currently
fails, while SAME-PROCESSproofsandcontinuationpass. Preserve thisboundary.

SparseFinalwasNOTported tocleanreplay. Its frozennoisy-passRoutePlans mayneed
logging too; do not claimlatentsalone sufficientforsparsehistory. Layer-major
parallelprefill/checkpoints are onlyfutureideas, notimplemented.

## Restore benchmark diagnostic, urgent next check

- Script `scripts/benchmark_native_restore.py`; frozenv2source
  `/tmp/longlive2-restore-diagnostic.dFx4PB/checkout` @2b2c537.
- Reads prefixlog/originalfullhashes frominflightcase, loadsnativegeneratorONLY
  (IdentityT5/VAE, becauseencodedconditionsalreadylogged), samefullstrictweights.
- Compare rawpageable, rawbounded-pinned (one84MiBstage), cleanlog. Combinedpinned
  budget128MiB;5warmup,30blockedrandomizedrepeats. Actualmetadata/fullKVhashes
  checkedbeforetimingandafter. StatebytesnotprocessRSS; benchmarkholdsbotharchives.
- v1onGPU1failedinitialfullhash. Noresults/timingswereaccepted. v2onGPU0adds
  `initial_step_diagnostics.json`, andonfailure`failed_full_hash_diagnostics.json`.
- Potentialcauses notestablished: missingruntimecontext, cold-kernel/numerical
  behavior, inputlayout, ordevice-specificdifferences. Normalizedconfiglooks
  correct; noTF32globalsetter wasfound in activeimports. Inflightoriginal/replayed
  hashes ALLmatch. Need the actualfirstfailinglayer/step toguidefix.
- v1source `/tmp/longlive2-restore-bench.23QzHy/checkout` @78b5dc7; preservefailure.
- Nativecachedefault isalreadyABSOLUTE-RoPE positioned
  (`causal_model.py`~478–488`key_to_cache=roped_key`, relativeRoPEdefaultfalse).
  Neverreuse1.3Braw-Krebase assumptions.

## Current16case admission/storage batch

Design `docs/NATIVE_EPISODE_ADMISSION_AND_REPLAY_20260908.md`.
Code `native_episode_memory.py`; runnerflag`--episode-memory-mode`.

- none: nativebaseline, noarchive.
- raw_reveal: keep8completedpositionedKVframes40–47beforeaway, replacetheexisting
  global8prefixslots onceatfirstreturn96. PreserveoriginalabsoluteRoPEpositions.
- raw_away: same8uniqueNONresidentframes56–63(firstnonpinnedawayblock), samebytes.
  This avoids acontrolmerelyduplicatingcurrentresidentKV.
- log_reveal: sameadmissioncoordinates/SHA asraw_reveal, butstoreonlycleanlog
  through48, scratchreplayunderpastconditionsatreturn, copylast8reconstructedKV
  GPU→GPU, restorecurrentcaches/settings/RNG. NooriginalKVretained bythisarm.
- AllkeepnativeAttentioncapacity32. RawCPUcap3GiB; log64MiB. Ledgerseparates
  archiveD2Hpayload, demandH2Dpayload, D2DKV, capture/demandwall, archivepeak.
  Native tinycontrolcopies are NOT silentlycountedaspayload.
- Admissionis PRIVILEGED (source/returnchosenbyexperiment). Do not callit an
  autonomousonlinePareto method. Newqualityresultsarepending.
- Realfour-armgate `longlive2_native_episode_gate64_v1`,256×512explicitlowres,
  starts0/8/16/48,64latents/253pixels, seed20260904.4/4pass; allpre48latents/noise
  exact; raw/log FULLlatent/RGBandlogicalplanSHAexact; bothchangedbaselineoutputs.
  RawCPU/H2D377,487,360bytes; log9,175,104bytes; logD2DKV377,487,360bytes.
  Sourcesgate8–15relevant,24–31wrong, bothnonresidentatreturn.
- Gatesource070b98f0cb1c07c8d66579434455eb5236a37901 at
  `/tmp/longlive2-episode-probe.x1zlAg/checkout`. Batch-only217c39dunchangedrunner/
  module/configverified. FullCPUregression485passed/1skippedbeforebatch.
- Submissionwrapper `../scripts/submit_sprint_native_episode_batch.sh`.
  OfficialInferHubmanual/SKILL/templatewerereadfully; publicenvs untouched.
- Recoverytargetplanned `results/videos/sprint24h_20260907/longlive2_native_episode509_h_v1`.
  Collector `collect_native_episode_batch.py` handlesfullroot and`--gate`.
  A fullreviewscript stillneedsadding: reuse `review_native_cut_feasibility.py`
  helpers, validateeachfour-armgroup, compareownbeforeaway/return/latequarters,
  andavoidcomparingabsolutehardwaretimesacrossruns.

## Useful Dense memoryworkload now established

- `longlive2_native_cut509_h_v1`:4/4actualH800,2newscenarios×seeds20260913/14.
  Officialprefix`The scene transitions. `ONLYonfirstblockofnewshot; starts0/24/
 48/96,48awaylatents, full704×1280,128latents/**509pixels**.
- Bothtoy sourcesareparticularcolorfulpatchworkfigures; returnsareclearlydifferent
  robotdesigns. State13redbeads→emptyjar; state14redbeads→flower/vegetationinjar.
- All128framesoffinal32-latentawayinterval inspected(two64frameboardspercase),
  onlyflowers/grass andno target visible atthumbnailresolution. Thisfixes old
  “objectneverleft” evaluationproblem. Itisdevelopmentfeasibility, notmemorypass.
- Source/return/allboards andfullawaygrids:
  `longlive2_native_cut509_review_v1/INTERPRETATION.md`.
- Job`zhouhe08__longlive2_cut_memory_screen_Iter0__6b9a90755ebf` COMPLETE;
  source6b9a90755ebfb40f4689fdfaefa9aba7e549ed28, all27filesrecoverySHA-match.
  Do notresubmit it. Originalscene-cutgate48/189also COMPLETE.

## Other closed findings (do not repeat)

- Streaming957extension12/12 on4090, allnoise/routes/latent/RGBexact. Threeblocked
  repeats: motionmedian1.1325×, state1.1552×; nonegativepairinthisextension.
  FirstMP4packet~5.3–7.1s vs~333–355s. CPUrawhistory65.56GB stillunbounded;
  delivery~3.8–4.1s per12pixels, NOT16fps(.75s)real-time. Allquartersreviewed:
  motionface/bodydrift, statedownwardlevelvariations. Systempreservestheflaws.
  Metrics`streaming957_local_audit_v1.json`, `streaming957_delivery_review_v1`.
- PriorH200477factorial48/48 remainscombined1.16–1.20×medians, negativerepeats
  retained, directRoPEunderasyncnegative3/4groups. NeverpoolabsoluteH800/H200/4090.
- NativecontinuousLongLive2controls4/4H800509: originalsnevertrulyleave;
  duckreplacementdelayed, absencefails. NoourmethodbeatsLongLive2claim.
- Past-textcontrols4/4local477: seed09neverleaves; seed10cheapoldinstruction
  restatementrecoverscoarsered/boxhead/triangle/whitefeetbutwrongfineface/badge.
  Thismotivates generated-informationtests, not anautomatictext-memorymethod.
-1.3Bpre/postrecacheKVversionintervention6/6H200477technicalpass, wrongidentities
  bothversions; no stablewinner, stopped. Seeepisode_snapshot477review.
- Tetherexistingreviewclosed withoutnewgeneration. Tea0hands/lightarcs, tea1
  subjectleavesdespitecenterinstruction; automaticcyclistmasks2fail; manual
  rescue1pass/1trackingfail. NofairAdaCluster/SVOO/SCOPEranking.
- Originalcostmodel,conditionalgroup/representation/precisionvideogates remain
  negative/notpromoted. No newcausalthree-role algorithm oradaptiveKVOutvideo win.

## Artifacts and remainingdelivery

- Mainprogressupdatedthrough08:20only; updatewith09–11findings.
- `NATIVE_CLEAN_COMMIT_REPLAY_20260908.md` nowexists (possiblyuntracked), precise
  evidenceandlimits. `MENTOR_SYSTEM_KVOUT_TETHER_UPDATE_20260908.md` summarizesthe
  mentor3requests butneedslatest09–11appendix.
- Figures `system_insight_figures_v1` (H200factorial/firstpacket, boundedproducer,
  KVversions) andolder`insight_figures_v1`complete. Addthecleancutfailure/episode
  findings andrestorelatencytradeoff onlyafterrealresults.
- `audit_sprint_video_inventory.py`: retrospectiveexecutioninventory, NOTcountof
  independentexperiments. As-of06:55closed232/232withpayloadhashes. Itwasextended
  fornativereference/text/cutcohorts, butstillneedsregistrationofcleanreplay,
  inflightreplay, four-armgate andpending16episodecohortbeforenextaudit.
- Needfinalallstartedcaseledger (pass/fail/negative, no missing), latesttests,
  videos/index/sourceSHA, negativeledger, paperinsights anddiscussionhandoff by14:21.
  Preserveallfailed/invalidoutputs. Do not claimtheoverallfaster-bettermethodgoal
  complete merelybecausecomponents/testsfinish.

Memoryregistrywasused forIO/causal conventions (`MEMORY.md36–39`); finalneedsone
memorycitationblock. No memoryfilesmodified. No subagents unlessexplicitlyauthorized.
