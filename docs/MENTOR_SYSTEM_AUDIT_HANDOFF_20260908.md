# 导师系统分析已交付：等待用户过目

最新用户指令：先列硬件/带宽/算力/各阶段工作量与耗时、瓶颈程度，说明QOut/KVOut
与FA4，并提供Perfetto和直观阅读说明；**完成后先让用户过目，再继续研究探索**。
旧“持续探索/不要停”不意味着现在可以自动启动新算法或GPU矩阵。

## 用户入口

工作根`/home/zhouhe08/MyProjects/0904-longlive-system`，事实源：

- `results/metrics/mentor_system_audit_20260908/review_final/index.html`：主量化报告。
- 同目录`FA4_QOut_KVOut.html`：论文/算子、三个搬运层次、cycle模型与迁移边界。
- 同目录`Perfetto_阅读说明.html`：轨道、操作步骤、SQL和误读例子。
- `results/metrics/mentor_system_audit_20260908/LongLive_导师系统分析_Perfetto审阅包.zip`：
  约10.3MB，包含HTML/Markdown、图、CSV/JSON、来源锁与9份trace。

9份trace均用官方Trace Processor v58.2导入及SQL检查，无非零error，
`review_final/VALIDATION.json`与`SHA256SUMS.json`保存验证及文件SHA。
这是原生解析验证；没有声称截取过Perfetto浏览器UI截图。引导图按真实时间戳重绘。

## 关键结果

- 官方白皮书确认4090 BF16/FP32累加密集Tensor峰值165.2TFLOP/s，不是330；
  本机128SM、72MiB L2、100KiB shared/SM。H200的1979等规格带2:4 sparsity脚注。
- 本机256MiB pinned H2D26.687GB/s；CPU pack＋H2D8.446；2线程1GiB CPUcopy
  payload12.366GB/s。串行模型1/(1/12.366+1/26.687)=8.450，与实测接近。
- 本机BF16形状匹配GEMM约77–160TFLOP/s，详细含5warmup/30测量；CPU affinity
  对照不是严格NUMA membind，不因某个远端pageable更快而推广NUMA规律。
- Final39基线诊断：generation56.536s，关联GPU activity21.544s；核心Attention
  3.972s/约141.9有效TFLOP/s，FFN5.010s/约119.8；VAE范围8.675s。
- Final39 archive物化H2D2.678GB/.102s≈26.19GB/s；文本H2D11.363GB/1.110s。
  不能把全部H2D都当history搬运。CPU准备、索引/整理与发射/等待仍重要。
- 主FA2 launch实际255regs/thread、48KiBshared、128threads/CTA；资源上界为
  2CTA/SM、16.7%warp occupancy。不是测得occupancy，不能据此断言低算力效率。
- 当前streaming153另外一条trace：parent41.654s，GPU活动30.349s，generation/
  VAE kernels同时6.807s。该优化版本和batch基线不同，不混成一个速度表。
- Fixed-edge QOut/KVOut不自动改变QK/PV FLOPs。已有有限驻留KV-major减少真实
  H2D的证据与各H2D一次的kernel网格分别报告。没有新kernel/算法视频推广。

## 本次实际执行及缺口

- 来源：16个首批公共来源全部下载，13个算子/Perfetto来源中1个旧FlashInfer
  URL404，已保留并用新namespace补齐；版本、URL、SHA均保存。NVIDIA PDF以
  私有pypdf解析；Perfetto验证器为官方清单SHA，不改公共环境。
- 硬件校准：23个bandwidth/copy配置＋9个GEMM配置，全部技术通过。
- 全流程：Dense/Final各一条39-latent v2，均无采集latent对照精确一致。
  模块真实shape包含未融合LoRA A/B、FFN、cross-attn、T5与VAE算量。
- v1计费不采纳：metadata读取DynamicSwap.weight额外产生9,261,023,232 bytes
  H2D（168次），虽然latent仍相同；另有T5计数/分类问题。v1不覆盖，勘误在
  `PROFILE_V1_ERRATUM.md`。v2读取原始参数字典并有防副作用测试。
- Ncu硬件counter缺口保留。GPU copy字节、launch参数、静态occupancy上界和
  有效FLOP/s均不是DRAM/L2/SMEM事务或Tensor Core利用率counter。
- 两基线39诊断使用archive-run/cache、metadata=recompute、grouped_fa2、batch
  VAE。当前优化流水单独用已验证旧trace，不将旧CPU空隙比例冒充当前版本。
- 项目`tests/`全回归505passed/1skipped；最后13项相关测试再次通过。

## 代码与继续条件

核心采集器冻结`0689f7d`，工作树`/tmp/longlive-mentor-profile-v2.EJg0WJ/checkout`。
代码在唯一项目Git根`publish_repo/`、分支`longlive-system`，本次主要新增
`operator_work.py`、硬件校准、Perfetto导出、launch资源分析、报告生成及验证脚本。
旧44/102/正式与sprint结果只读；未训练；未修改公共驱动/环境；用户原有3个
未跟踪文件保留。没有新InferHub提交。不要重跑已完成诊断或正式矩阵。

下一步先等待用户审阅本材料。之后才讨论核心机制、统一强基线、独立长视频与
消融。H200更大驻留、LoRA融合/图化、kernel资源改造等仅作为审阅后的候选，
没有自动授权新一轮矩阵。
