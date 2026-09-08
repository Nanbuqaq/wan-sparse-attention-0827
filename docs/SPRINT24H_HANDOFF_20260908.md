# 24h研究冲刺交接：从已关闭证据继续，不重投

## 先读

1. [本轮研究结论](SPRINT24H_RESEARCH_REPORT_20260908.md)
2. [导师system/KVOut/Tether三项更新](MENTOR_SYSTEM_KVOUT_TETHER_UPDATE_20260908.md)
3. [Clean-commit与检查点恢复证据](NATIVE_CLEAN_COMMIT_REPLAY_20260908.md)
4. [时间序进度](SPRINT24H_PROGRESS_20260907.md)：里面旧时刻的pending不代表实时状态。

工作根`/home/zhouhe08/MyProjects/0904-longlive-system`，唯一Git根`publish_repo`，
分支`longlive-system`。9月7日原正式计划已结项；本轮为随后独立24h探索。
主目标仍是因果流式长视频更快、更好，尚无完整算法—系统共同赢家。

## 已关闭，禁止重复提交

- H200 episode task `zhouhe08__longlive2_episode_admission_storage_Iter0__217c39de6896`：
  **16/16**技术与配对契约通过，实际H200、native704×1280/509pixels。
  Raw/log相关KV同admission SHA、全latent/RGB逐位相同，归档张量约90.10×差异。
  视审四组均无稳定整体质量成功，相关信息只在部分seed取回；global方案不推广。
- Native H800 cut screen `longlive2_native_cut509_h_v1`：4/4，确有目标缺席，
  玩具换设计/红珠丢失，为有效开发motivation；非新方法胜出。
- Placement lowres `longlive2_native_placement_gate64_v1`：**6/6**技术/配对通过，
  生成SHA`12fde4fad378fa9380ceaa89e7dc0fa91376e125`。两个baseline的away仍有目标，
  因此标记workload-invalid，不进入质量均值/Pareto。只有4次未启动资源误拒绝
  得到attempt2补跑，原失败log与首两条成功保留，没有重跑成功case。
- Cold replay v1/v2 fullhash失败、v3离线recipe诊断、v4记录recipe后新进程精确
  恢复：全部保留。v4物理GPU1数据不与v5物理GPU0绝对时间混合。
- `native_restore_benchmark_v5_hybrid`：冻结SHA
  `ce7dfa1bbabc809a3ef051ecb7fd7aef206ec186`；六路径全部full original KV hash
  通过，各5warmup/30blocked随机重复。纯日志22.282MB/1.3932s；16帧KV+
  tail2,653.554MB/1.0765s；raw5,284.823MB/.5273s（median）。含pack pinned
  median .5694s但p95 .8210s优于pageable1.1404s。仅warm恢复，不含创建/驱逐/
  冷盘网络/实际RSS；不是视频加速或自动冷热scheduler。
- Blackwell task `zhouhe08__longlive2_episode_5kpro_replication_Iter0__4263845f2ccb`
  在CPUprep因Torch/Triton `AttrsDescriptor` 不兼容失败，0GPU生成。8门禁与
  条件16长视频均未启动。不得记为方法质量失败；原H200任务没有被取消或重投。

## 系统/Tether仍然成立的边界

- H200477 48执行组合中位1.16–1.20×；新增4090 957 12执行，motion1.1325×、
  state1.1552×。相应ordered routes/latent/RGB精确相同，negative重复保留。
  首MP4包约5.3–7.1s不是client显示；3.8–4.1s/12帧不是16fps。CPUarchive仍
  65.56GB，系统没有修复motion身份漂移/状态非单调。
- KV-major在有限residency的同逻辑图上实际减少重复取数；producer/graph后
  有真实H2D/kernel重叠，但没有生产history-D2H/H2D-overlap视频收益。
  72点驻留数据流reference中Q-stationary家族≥90%winner，未推广adaptive。
- Tether只保留效果与机制证据，不继续追求完美two-pass复现。茶壶/骑车无稳定
  整体质量赢家，自动mask失败和手工oracle分开；没有公平新AdaCluster/SVOO/SCOPE排名。
- 新成本模型、复杂utility、conditional grouping及旧KV版本等负门禁不翻案。

## 终态审计与入口

事实源均在`results/metrics/sprint24h_20260907/`：

- `discussion_index_v1/index.html`：图表、四组视觉链接、系统与Tether入口。
- `video_inventory_final_20260908.json`：**278/278有终态、remaining0**，
  276技术pass/2技术fail，包含重复/短门禁；所有video/latent payload SHA。
- `native_boundary_terminal_audit_20260908.json`：恢复v1–v5、Blackwell prep
  失败、四次资源检查误拒绝及最后两份语义review的完整性审计。
- `longlive2_native_episode509_review_v1/INTERPRETATION.md`及
  `semantic_verdicts.json`：16条完整视频审查。
- `longlive2_native_placement_gate64_review_v1/INTERPRETATION.md`：技术通过、
  低分辨率任务无效；不是全面否定shot记忆。
- `native_hybrid_restore_figures_v1/index.html`：新的容量—median/p95折中图。
- `cpu_regression_final_delivery_v2.log`：项目`tests/`最终回归。
  先前未限定目录的pytest误收第三方源码测试而collection失败，旧log保留。

## 下一轮最小决策，不是直接启动新矩阵

先与用户讨论：是否把原生分镜任务上的信息激活/场景隔离，作为下一轮算法主线。
如继续placement，必须回到原生分辨率和有效absence任务，不能消费短gate当质量。
如继续存储，测检查点创建、跨进程/节点迁移、并发干扰和真实重访/恢复deadline；
目前只证明精确state primitives和有限warm tradeoff，没有无界历史释放策略。
如继续KV-major，先让实测CPU供给和消费组织进入完整模型关键路径再扩视频。

## 资源与源码保护

最后GPU实验均已结束；接手仍需实时核对GPU/进程/InferHub，而不是按旧检查点重启。
使用`run_on_free_gpu.py`物理锁；串行lane持有外层锁可避免进程退出后的utilization
采样滞后误拒绝。两个冻结worktree仍保留在`/tmp/longlive2-placement-gate.5dXNmQ/checkout`
和`/tmp/longlive2-hybrid-restore.3tMfNd/checkout`，`/tmp`不保证持久，Git/结果为事实源。
本地调度脚本在工作根`scripts/`，不把绝对环境路径伪装成公开可移植入口。

不训练；旧44/102/正式结果只读；保留用户原有未跟踪
`scripts/audit_retrieval_divergence.py`及0400/0500检查点。新增Blackwell输入bundle
含原权重硬链接，不可编辑/chmod。不得修改公共平台环境或绕过GPU计数器权限。
当前不得主动spawn子agent，除非用户或适用指令明确授权。
