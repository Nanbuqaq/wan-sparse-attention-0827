# Observer-only native startup breakdown

Native full cases repeatedly report a constructor/load span around150s, with
substantial run-to-run variation. Source inspection shows model construction
and repeated parameter initialization precede loading complete released weights;
T5 is first allocated/transferred FP32 then cast BF16. These are hypotheses for
the time, not yet measured attribution or proof of a loading bug.

Run one source-locked startup-only observer after current video tasks release
their GPUs. Preserve all initialization and exact strict checkpoint operations;
no generator forward, video, training or optimized loader. Nested host scopes:
DiT from_config, T5 constructor, VAE constructor, init functions, torch.load,Module.cuda host span,
T5 BF16 cast, strict generator load, final GPU readiness. Init metadata includes
device/dtype/logical tensor payload, NOT measured CPU/HBM traffic.

Report inclusive and exclusive host durations without summing nested parents;
file load combines filesystem/cache/deserialization work. Page-cache state and
observer overhead are not controlled, so this single run cannot establish a
startup speedup. If redundant initialization dominates, a subsequent strict
checkpoint/derived-buffer/numerical gate would be required before changing it.
