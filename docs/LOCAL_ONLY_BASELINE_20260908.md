# Same-backbone full-local baseline

An important control uses the same Wan1.3B/LongLive+LoRA weights and common direct-outputRoPE, but memory_size0 and the entire12-frame GPU local cache. It creates no historical CPU archive and performs no history H2D. Its original identity may differ from RAG because its context policy differs; score return consistency against its **own** reveal appearance.

The adapter's memory_size0 path supports native_block/block64 identities. At history_density1, all local tokens are dense. The former implementation still computed/sorted sparse scores and gathered all tokens; that unnecessary work is now bypassed at100% density. Small and full-shape realGPU gates compare against unchanged upstream local attention and verify bitwise output/cache equality over five calls, with zero archive and history transfer.

This is not LongLive2, and not a cross-backbone quality claim. It is the strong same-backbone baseline needed to judge whether old-history complexity earns its cost.

Planned control set: original toy event schedule seeds20260909/20260910, plus the frozen duck/absence controls at their existing seeds, each one local-only477 video. All generation/decode/encode costs remain charged and memory/transfer state is audited. No old experiment is overwritten.
