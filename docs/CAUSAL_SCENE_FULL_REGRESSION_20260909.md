# 自动场景选源的5条完整回归

bd3d930短GPU gate已过：自动归档ends8/16/48，自己选择8–15在48安装，整段latent/decodedRGB
与旧指定源recent gate相同。没有人工source fallback。archive额外D2H3×，已计费。

固定5条原生509：bead-fill seed13、settled seeds19/20、toy seeds13/21；所有输入、参数不变。
包含已知toy21身份结构失败，不能只回归正样本。5条都与旧privileged recent版本做整段输出等价。
这不是新增5个独立质量成功，也不证明对未见prompt的检索能力。

GPU0依次bead-fill13、settled19、toy21；GPU1依次settled20、toy13；失败不阻断其他case。
既有视频不重跑为对照，不覆盖已有结果。本阶段不修改floor0.80、margin0.05、8GiB FIFO或cue规则。
source与target帧只用于生成结束后的验收，不进入当前输入/descriptor selector。
归档3个已完成scene，完整archive字节约7.785GB，不能和指定单源2.595GB成本混为一谈。

验证完整latent/RGB、source选择、真实原生pin、H2D与CPU归档预算，并报告完整generation时间。
未profile的HBM/关键路径等待不从墙钟或CUDA API调用中编造。
