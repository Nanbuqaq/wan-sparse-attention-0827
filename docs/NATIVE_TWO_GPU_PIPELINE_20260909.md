# 真实双卡流水：先同输出门禁，再测时间线

原生VAE增量解码已在低分辨率和704×1280/509帧逐位通过。本阶段接入真正DiT生产。
不修改原生模型或权重，不调用原生WanVAE缺失的cached_decode，不切换LightVAE。

- 保留DiT return_latents=True；仅在当前chunk的clean commit完成后取最终latent。
- 显式GPU0生成、GPU1解码，双卡原子物理锁；没有两把锁的运行拒绝启动。
- 两个固定pinned latent槽；source snapshot→source ready event→D2H→CPU ready→H2D→GPU1 VAE。
- 槽一直归消费者所有，直到该chunk完成解码/编码；不得在DMA或CPU哈希读取时覆盖。
- 一个固定pinned pixel buffer，按已完成pixel group送增量sink，不收集整个视频GPU/CPU像素列表。
- 增量latent SHA必须等于最终返回的完整latent；原生视频RGB SHA也必须等于参考。
- 消费者异常应解除生产者等待并保留partial结果，不能后台线程死掉而前台永久等槽。

固定两个配置：serial双卡每chunk等消费完成；overlap双卡允许有界前推。其余路径与资源一致。
先各跑一个64-latent真实门禁，与旧未改生成的native64控制核对；不据短视频做质量结论。
CPU host trace可以在Perfetto看顺序/背压，但不是GPU overlap证明；保留NVTX用于后续Nsys/CUPTI相关性。
CUDA事件记录同设备stream跨度，不跨设备相减，也不把有重叠的组件时间简单相加。
完整计入VAE预放置、pinned池、latent D2H/H2D、pixel D2H、编码、首包和完整wall；客户端显示时间未测。
