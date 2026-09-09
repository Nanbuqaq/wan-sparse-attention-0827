# 本轮研究进展：可验证收益与未解决问题

完整研究目标尚未完成；下列区分系统收益、机制信号和明确负结果，不作SOTA排名。

## 已闭合的系统结果

- 原生5B双4090、509帧：CPU输出完成线程在3对重复中把完整交付中位数98.95s降到84.81s，各对下降12.39–14.69%。6/6实际完整latent/RGB及buffer ownership通过。
- Nsight确认约92.15%的CPU输出范围与GPU1 kernel重叠，kernel服务量基本不变。不是“异步API已调用”的推测，也不是H2D字节减少。
- 首包没有稳定改善；额外一条CPU完成线程和43.254MB pinned输出缓冲单列。只验证这一工作负载/双卡，不声称跨硬件或优于两条独立单卡任务的吞吐。
- 初始化145s的诊断中，CPU初始化函数约97s、torch.load约40s。跳过待完整checkpoint覆盖的Parameter初始化，通过253/509帧实际完整输出等价；约52/53s只是两次门禁观察，不是生成阶段加速。
- VAE channels-last有数值变化，30对解码测速仅约3.33%下降，未达到10%深入门槛，停止该分支。

系统事实入口：NATIVE_PIXEL_COMPLETION_REPEAT_RESULTS_20260910.md、NATIVE_PIXEL_COMPLETION_TRACE_RESULTS_20260910.md、STRICT_CHECKPOINT_INIT_RESULTS_20260910.md。
真实Perfetto：工作根results/metrics/memory_activation_20260910/native_pixel_completion_profile509_v1/audit/native_two_gpu.perfetto.json.gz，官方解析通过。

## 算法实验与边界

- 木箱源有效、确实离场，两seed在相同source/bytes下，原位置与recent位置均未恢复开盖状态。不能从红珠结果推广成通用状态方法。
- 只补过去状态文本可生成粗略开盖/蓝布，但物体细节漂移、一个seed后段出现手来合盖。
- 相同文本下加KV，两个seed的箱体/布料更接近source；一个仍后段合盖。因此保留2×2交互信号，不称稳健长期方法。
- 保留原source pin的追加两例，通过104latent/413pixel前缀、实际pin和驻留K/V采样审计，但仍未消除失败seed的合盖。可用性不是充分解释。
- 这些是同一任务家族/两seed的控制实验，不是把多次生成当作多个独立holdout样本。没有换seed、改措辞或调delta追正结果。

两seed并列的2×2图：工作根results/figures/chest_condition_history_factorial_v2/index.html。
机制事实入口：CHEST_CONDITION_MEMORY_FACTORIAL_RESULTS_20260910.md、CHEST_SOURCE_PIN_LEASE_RESULTS_20260910.md。

## 正在补的证据

匹配每个head/query的source总Attention概率后，时间重定位仍改变组内读出；该结果只来自隔离teacher，不提供在线score。
LayerRecall已提出当前状态检索与层选择；不能把这些已知问题当独创新意。此前0/14/29采样不覆盖其memory-sensitive层，故启动全30层离线诊断：返回首/末chunk、三阶段，每case180记录，保存紧凑组统计而非完整K/V。
此批生成必须与已有hybrid完整输出相同，FP32各行单独过门禁；没有训练、不据单一profile直接冻结layer allowlist。实时状态看ACTIVE_HANDOFF_20260910_RESUMED.md。

发布到公开GitHub的明确授权尚未获得，因此未push或新投InferHub。此前所有旧结果、负结果及用户原有文件均保留。
