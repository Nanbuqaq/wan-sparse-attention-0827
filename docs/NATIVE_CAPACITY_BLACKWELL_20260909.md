# 独立Blackwell容量基线，不混用旧编译器轨迹

原H20仍保留de72255任务，H节点心跳中断后由平台退回pending，未重复提交。
这批在5kpro上补原生5B容量对照，使用私有Triton3.3.1＋TORCHDYNAMO_DISABLE=1，
BF16/FA2、adaLN16/1。没有修改公共环境，也不声称与Triton3.2逐位等价。

背景：3.3.1已在真实4090跑完整视频，但跨3.2精确门禁失败，第一chunk relativeL2已0.207；
所以不能直接借用旧轨迹作控制。这里先做同版本、同卡的实际分配等价门禁。

冻结4 lanes：toy/bead × full seeds20260915/20260916。每lane先运行两条
64 latent/253 pixel硬件gate（gate seeds20260917/20260918）：原生positive+negative
与positive-only，要求全部noise/latent/RGB相同、KV张量减半。gate失败则不启动该lane完整视频。

通过后每lane运行positive-only的local32、local128两条原生704×1280、128 latent/509 pixel。
全批8短gate＋最多8完整容量case，不加入新召回方法。窗口变化允许自身轨迹改变；
同seed的noise必须匹配。预期真实positive KV为10.381/41.524GB，以实测峰值判fits，不做fallback。

全部结果按该硬件/runtime独立报告。审查各自source、away最后128帧及首/晚返回；
如果大窗口导致目标从未离开，记切换/任务完成失败，不把它的持续可见当记忆成功。
这不是正式holdout或方法Pareto；大容量不是预设输家。

一次冻结提交，CPU prep不占卡；每lane实际GPU生成、失败独立保留，不重试已成功case。
