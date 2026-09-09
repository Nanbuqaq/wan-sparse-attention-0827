# 原生VAE增量解码门禁

已核对当前`vae_type=wan`绑定`WanVAEWrapper/_video_vae/WanVAE_`，该类没有`cached_decode`。
原生pipeline的streaming路径却调用此接口；LightVAE有实现，但不能悄悄换VAE/权重来通过。
`return_latents=True`还会关闭原生streaming分支，故仅翻配置不能构成已实现的流水线。

新适配器保留原生decoder逐latent调用、first_chunk及feat_cache规则，只跨输入chunk保持缓存。
conv2为1×1×1；每次产出一个已完成pixel group，不做不断增长的整视频GPU cat。
不同chunk长度仍可能触发CUDA数值差异，不能只凭卷积时间点独立性宣称逐位等价。

先重放已保存的64-latent低分辨率输入，原生batch输出须复现旧记录RGB；
再检查chunk8/7/13的完整253像素帧、浮点误差和RGB SHA。比较对象是原生BF16解码，不是FP32模型。
CPU逐像素误差/哈希计入本门禁墙钟，不用它声称性能改善；不生成新DiT视频。
只有通过后才做原生分辨率、真实两GPU供给/解码/输出时间线及端到端计费。
放弃未消费完的iterator会使流失效，不能悄悄跳帧；新视频必须显式reset，不能泄露前一视频缓存。
