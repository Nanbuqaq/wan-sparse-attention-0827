# 原生加载适配层：慢在哪里

一次真实observer诊断总加载范围145.258s，无生成forward和新视频。扣除嵌套重复后，
CPU参数初始化函数范围97.318s（约67%），torch.load40.089s（约27.6%），显式
Module.cuda主机范围2.340s。其余为模型管理、拷入参数、cast等主机范围。

三个权重读取范围分别为T5 3.189s、VAE8.688s、generator28.212s；对应文件11.362、
2.819和10.000GB。缓存冷热未知，不据此推断磁盘/网络带宽。Module.cuda含分配/上下文/
复制等待，不是PCIe DMA服务时间。初始化logical字节可能重复或来自meta，不是DRAM计数。

这解释的是我们“from_config构造→严格加载完整merged checkpoint”的加载适配路径；
不能扩大成LongLive2算法本身必然慢。其随机初始化随后被完整参数覆盖，值得独立验证是否
可省；但尚未证明修改后输出一致或获得可重复的加载加速。

`startup_host.perfetto.json`可查看主机嵌套范围，无GPU设备活动；不要将它与此前真实双设备
CUPTI trace混为一谈。完整json保留每个事件、exclusive/inclusive时间、来源SHA和严格加载状态。
事实：`results/metrics/memory_activation_20260910/native_startup_profile_v1/`。
