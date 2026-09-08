# 24小时探索：流式生成的信息、组织与生命周期

目标仍是因果流式长视频“更快、更好”，不是把每条路线做成正结果。本文件接续
9月7日已关闭的原正式计划；本轮开发实验不重新把已消费正式集当作 holdout。
以下实验、短放置对照与混合检查点恢复测试均已获得终态；原始失败保留。

## 1. 本轮最值得讨论的三个发现

### 信息记忆必须用“生成出来的信息”来考，而不只是复述 prompt

旧连续镜头任务经常没有真正离开主体，不能用于证明跨离开区间的记忆。
新的原生 LongLive2 分镜协议使用官方 scene-cut 前缀，玩具具体外观、罐中实际
珠子状态由模型生成，返回文本不重述这些具体内容。两任务双 seed 的离开末段
全部128帧接触表没有目标；返回时两玩具换设计、两罐红珠丢失。

这是更有效的开发测试，不是新的方法成功，也不是 LongLive2 总体能力排名。
过去文本重述可恢复粗类别/颜色，却不保证保留原生成细节，需作为便宜的独立
对照。今后质量应分别看原生成信息、当前指令、新场景，以及长期状态一致性。

### “取回相关 KV”与“正确使用它”是两件事

16条 H200 四臂实验已经消除了存储重建误差：raw/log 同样准入、全视频逐位
相同；无关 KV 控制同字节且不重复 resident 内容。相关记忆在一个玩具 seed
带回原 cyan 面孔，在一个状态 seed 带回红珠，但仍有花瓣混合、碎片与绿色珠子。
四组都没有稳定整体质量成功，global-prefix 替换不推广。

最后六条256×512短放置对照全部技术通过、源内容与字节配对成立，但两类别
baseline都未真正让目标离开，且有明显画面伪影。因此只能记录放置/被替换上下文
影响结果，不能把shot较少花朵混合宣称为长期记忆改进；不再扩新长矩阵。

因此下一步不应只升级一个相关性分数：还须分离长期身份/已提交状态和短期场景
上下文，明确在哪个 query、层、生成阶段激活，何时退出。分组只有与使用协议
和生命周期结合才可能有价值。这是由受控失败提出的假设，不是已验证的三角色
自动算法；目前也不能断言污染完全由某一个位置/RoPE因素造成。

### KV 可以是热表示，已提交生成日志可以是冷表示

原生 clean commit 的实际 BF16 latent、编码条件、位置/pin指令与数值执行配方，
可精确重建派生 KV。已实测真实释放—重建—继续生成，未来 latent/RGB不变；
新进程读取已记录配方也通过全部原KV哈希，未搜索配置或拿teacher选择参数。

H200 episode中保存同一8帧记忆：raw归档2,595,225,600 bytes，前缀日志
28,803,264 bytes，约90.10×。但4090独立进程warm恢复中，raw pageable
median0.3057s，bounded pinned（含pack）0.5006s，日志重建1.3655s。
这是容量—恢复成本的真实取舍，不是“小90倍所以快90倍”。前缀日志仍增长；
不包含模型权重、不等于进程RSS、不代表有界无限历史，也未证明稀疏Final可只
凭latent精确重放。

最后的检查点＋日志尾部测试已通过全部原KV哈希。新同卡GPU0配对表：

| 表示 | 保留张量MB | 恢复median秒 | p95秒 |
|---|---:|---:|---:|
| 纯日志 |22.282|1.393|1.397|
| 8帧KV检查点＋日志尾部 |1,337.918|1.239|1.392|
| 16帧KV检查点＋日志尾部 |2,653.554|1.076|1.344|
| 24帧KV检查点＋日志尾部 |3,969.189|.883|1.284|
| 完整raw pageable KV |5,284.823|.527|1.140|
| 完整raw bounded-pinned（含pack） |5,284.823|.569|.821|

每路径5预热、30随机交错重复。不能把上文GPU1旧测试时间拼进这张表。
中间容量—median折中成立，但检查点的p95改善小于median；pinned的median稍差、
p95反而更好。流式系统需要看deadline与尾延迟，不能只按平均带宽选路径。
这里只测warm恢复，不含创建/驱逐、背景干扰或冷盘/网络，且未测实际RSS节省。

## 2. 系统层：哪些已经成为完整收益

| 层级 | 已有证据 | 严格边界 |
|---|---|---|
| 全流程分析 | 冷启动/权重、文本、检索、route、pack/H2D、RoPE/group整理、Attention、FFN/cross-attn、VAE和编码分开 | 主机范围和GPU活动不可简单相加；Ncu计数器仍无权限，不能证明全程HBM带宽饱和 |
| 同route执行组织 | metadata复用、resident group描述符、减少物化、严格per-chunk cache进入视频 | Dense和Final均获得通用优化机会；不把实现基线差异称新admission |
| 生成—VAE流水 | H200477 48执行，组合配对中位1.16–1.20×；4090 957再12执行，motion1.1325×、state1.1552× | 全部相应route/latent/RGB精确相同；H200有negative重复，directRoPE在async下3/4组negative |
| 流式交付 | 957首MP4包约5.3–7.1s，而串行约333–355s | 服务端mux不是client显示；约3.8–4.1s/12帧，不是16fps实时；CPU archive仍65.56GB |
| 有限显存KV中心调度 | 同逻辑图高共享cap4：Q-major288copies/108MiB，KV-major96copies/36MiB | 对应完整replay约102.29→76.49ms；不是完整视频，旧页准备不在该replay内 |
| 含CPU pack的双buffer供给 | Q4680 eager consumer几乎无益；CUDA graph后serial16.82→async13.63ms | Nsys测到2.652ms H2D/kernel重叠；独立producer无额外收益，未进生产视频 |

完整解析入口：[导师三项要求更新](MENTOR_SYSTEM_KVOUT_TETHER_UPDATE_20260908.md)、
[全流程图及原始审计](SYSTEM_FULL_FLOW_AND_REPRODUCTION_AUDIT_20260907.md)。
这些结果不是“纯memory-bound”的单标签结论，而是阶段相关的混合瓶颈。

## 3. 算法与Tether：没有被系统正结果覆盖的负结论

- Tether已使用官方CF+AE、无LoRA、长视频reference/Tether/neutral控制并审查。
  茶壶有额外手/光圈或后段失去居中主体；自动骑车mask两seed失败，手工oracle
  仅救回一seed。无稳定整体质量赢家，也没有同条件AdaCluster/SVOO/SCOPE新排名。
- Tether的价值在于区分信息并控制其作用；two-pass与全视频mask不是我们的终点。
  neutral零bias也有长期轨迹/检索漂移，不能把全部变化归因于semantic bias。
- 条件group在capture改善部分层的output误差，但增大的union、实际成本与状态
  视频结果未形成双类别优势。更少scheduled pairs不是更低物理开销的保证。
- 旧cost model仍negative；Top-p、复杂utility、precision/representation、
  pre/post recache旧KV版本等门禁没有得到可推广的共同赢家。详见时间序进度，
  不以一次颜色恢复或kernel加速翻案。

## 4. 从证据出发的下一轮决策

1. 以新生成信息任务保留baseline、过去文本、相关/无关记忆三个可解释控制，
   先测“激活相关信息但不过量携带旧/无关场景”的最小机制；不要立即铺满新矩阵。
2. 同一逻辑记忆分别使用raw、检查点＋日志、纯日志，再按照重访频率和恢复deadline
   评估冷热表示；测创建、迁移、后台干扰与实际RSS，不能仅凭warm恢复选系统策略。
3. KV-major继续以有限residency、真实CPU供给和消费调度为条件推进；只有端到端
   暴露等待改善且保住质量时进入视频，不由72点kernel表自动推断部署收益。
4. 方法改变后用新的独立开发/验证集冻结；自动causal selector、生命周期机制和
   长期预算约束都需真实闭环，不能把本轮privileged episode admission当成完成。

当前最可信的论文motivation是：**长期生成的信息价值、激活方式、存储表示与
执行组织需要一起设计；更多历史或更多异步并不自动带来更好/更快。**
尚不能声称已经获得全面优于LongLive/LongLive2的算法—系统共同赢家。

## 5. 证据入口与原始边界

- [完整进度和全部阶段的正负结果](SPRINT24H_PROGRESS_20260907.md)
- [Tether长视频审查](TETHER_LONG_VIDEO_FINDINGS_20260908.md)
- [Clean-commit原理、数值配方和完整恢复证据](NATIVE_CLEAN_COMMIT_REPLAY_20260908.md)
- `../../results/metrics/sprint24h_20260907/system_insight_figures_v1/index.html`
- `../../results/metrics/sprint24h_20260907/native_memory_figures_v3_recorded/index.html`
- `../../results/metrics/sprint24h_20260907/native_hybrid_restore_figures_v1/index.html`
- `../../results/metrics/sprint24h_20260907/longlive2_native_episode509_review_v1/INTERPRETATION.md`
- Blackwell额外复制批次在CPU prep因Torch/Triton `AttrsDescriptor` 不兼容失败，
  没有GPU生成启动；原H200批次没有重投，失败receipt已SHA回收。

执行inventory包含重复和短门禁，不是独立科学实验数量；技术pass不等于语义pass。
保留所有原始失败、旧结果、用户未跟踪文件和未推广路径，不训练、不跨硬件混报。

截至最后六条短对照结束，命名sprint视频inventory **278/278有终态、missing=0**，
所有视频/latent payload SHA已计算，无未注册目录。原正式与Tether另有各自审计，
不在278中重复合并。事实源：`video_inventory_final_20260908.json`。
其中276技术pass、2技术fail；语义negative/无效工作负载单列在视审，不伪装成
276个质量成功。最后native恢复与Blackwell/启动失败链审计也通过，见
`native_boundary_terminal_audit_20260908.json`。项目`tests/`目前498passed/1skipped；
最后图表断言补充后再次验证，终态记录见交接。
