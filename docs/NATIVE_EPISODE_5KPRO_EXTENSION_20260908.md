# Independent72GBBlackwell extension during H-pool queueing

The originalHbatch remains intact andqueued; do not cancel,resubmitoroverwrite it.
This is a separatelyidentified cross-hardware stage, not another execution ofthe
samefrozenHjob. Absolutehardwaretimes willnot bepooled. Current5kprohost reports
8freeGPUs; earlierverifiedhardware isRTXPRO5000**72GB**, not48GB.

Use the existingverified privateTriton3.3.1 overlay andimmutablehardlinkednative
weights in a NEWinputroot. Do notinstallintopublicenvs ormodifyhardlinkedmodel/
manifestfiles. Record fixednativeadaLN16warps/1stage inallarms toavoid unrecorded
autotune-numerics changingpre-interventionprefixes.

Onefrozen8GPUbatch: eachGPUhasadistinctscenario×mode. First8realhardwaregates
(two scenarios×fourarms,64latents/253pixels attechnicalgateresolution). Only if
bothgroupspassprefix/raw-logoutput/admissionSHA contracts, run16full509videos
(same2developmentseedsperlane). Thegateandconditionalfullstage aretogether;
failurepreventsfullvideo promotionandreceivesexplicitnot-launchedterminalrecords.

Thisextendsplatformrobustness andusesavailablecapacity. Itdoesnotpromoteprivileged
admission asautonomous, doesnotclaim fasterqualityacrossbackbones, anddoesnot
removeanynegativeH/localresults.
