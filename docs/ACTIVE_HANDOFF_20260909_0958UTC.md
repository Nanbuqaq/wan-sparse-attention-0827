# 当前继续执行入口（2026-09-09 09:58 UTC / 17:58 CST）

用户要求持续论文探索直到完成或手动停止；当前goal工具已active，无token budget，不要误标complete。
不要spawn agent。两张4090均授权，运行必须经`run_on_free_gpu.py`物理锁。

## 权限与工作区

- 工作根`/home/zhouhe08/MyProjects/0904-longlive-system`；Git根`publish_repo`，分支longlive-system。
- 运行代码冻结`f3da2f5509a169980cfec385e5f3213a2a1bb747`，最新CPU **640pass/1skip**。
- **禁止再次push或换渠道绕过**：向`https://github.com/Nanbuqaq/wan-sparse-attention-0827`发布被权限审查拒绝，等待用户明确授权；最后成功push为7b18592。新InferHub提交暂停。
- 不训练、不改旧44/102/957结果或第三方代码/原权重，不reset/clean。
- 保留用户旧未跟踪文件：0400/0500 checkpoint以及`scripts/audit_retrieval_divergence.py`。
- 服务器普通`date`/ls/nvidia-smi是CST；UTC用`date -u`/clock工具。不要再把显示时间误标UTC。

## 当前唯一GPU任务——不要重启

`results/videos/memory_activation_20260909/blue_positive_stop509_v1/`

- GPU0 / session **94420**：seed20260923，positive-stop revisit → positive-stop visible-control。
- GPU1 / session **70547**：seed20260924，同两条。
- 外层脚本：`scripts/run_blue_positive_stop_20260909.sh`（工作根scripts，不在Git子根）。
- 冻结worktree：`/tmp/longlive-blue-wording.H0quaY/checkout`@f3da2f5。
- 已dry-run、带物理锁启动；基础架构记录`results/infrastructure/local/blue_positive_stop_v1`。
- 只在latent32把`The painter and roller have left the view.`替换为
  `Only the canvas and its wooden easel are visible in the quiet studio.`。
  其他hold文字、颜色不重述、所有cut和seed不变；仍Dense-only，selector不动。

完成后在publish_repo执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/review_blue_canvas_screen.py \
  --positive-stop \
  --root /home/zhouhe08/MyProjects/0904-longlive-system/results/videos/memory_activation_20260909/blue_positive_stop509_v1 \
  --reference-root /home/zhouhe08/MyProjects/0904-longlive-system/results/videos/memory_activation_20260909/blue_canvas_screen509_v1 \
  --output /home/zhouhe08/MyProjects/0904-longlive-system/results/metrics/memory_activation_20260909/blue_positive_stop_review_v1
```

新目录必须不存在。脚本验证actual pre32/前125 decodedRGB等于旧措辞、同seed两分支pre64/前253像素一致。
逐帧看source56–63（pixel221–252）是否真停止、蓝/白区域是否有效；完整away与返回也需审查。
若只改文字就改善，归因文字；若仍失败，保留negative，不强行当干净state holdout。
蓝画布代码仍拒绝memory；只有source可行性审核并冻结SHA之后，才决定独立方法验证。

## 已完成，不要重复

统一入口：`results/metrics/memory_activation_20260909/discussion_index_v1/index.html`。
导师系统/硬件/FA4/QOut/KVOut/Perfetto包已早先交付：`results/metrics/mentor_system_audit_20260908/review_final`，不重做。

### 1. 历史K位置

- seed13四格源内容×original/recent：3新视频＋旧related-original，技术全pass。
  related-recent恢复红珠；两away污染；晚段仍倒入液体。审查`key_position_review_v2`，解释/判读在v1。
- 新seed21四格全pass，重复上述状态类别信号，仍新动作。`key_position_seed21_review_v1`。
- phase-only16/age-only48两新原生视频＋复用original0/recent64，四格pre96/pixel0:381精确。
  phase-only空罐，age-only少量变形橙红内容，完整recent恢复红珠但仍新动作。
  `key_components_review_v1`。两个分量加在同一时间旋转，不是两种独立编码。
- 变换只针对已保存BF16 K的44个时间实通道，空间K和全部V不变；不是恢复pre-RoPE原K，也不是layout等价。
- 相关source40–47→virtual88–95/currentphase24，delta64；away56–63 delta40。
- 每完整原生source H2D 2,595,225,600 bytes，原生active KV32，source/target目前多数机制实验是指定的。

### 2. 较静止红珠与身份边界

- `settled_memory509_v1`4/4pass（seeds19/20×original/recent），复用2个Dense；actual pre96/前381RGB配对。
  两recent红珠恢复，原位置/Dense空罐。两recent全部128返回帧已看，无此前明显倒入/搅动，
  仍有形状/玻璃/填充外观缺陷，source也不是严格运动零。`settled_memory_review_v1`及`settled_memory_return_pages_v1`。
- toy13三新＋一旧控制，recent明显更接近原脸/拼块/身体，末段有人手；wrong源花朵污染。
  `key_position_toy13_review_v1`。
- toy21 original/recent两条全pass，但两者都改写肩部/身体，recent仅部分脸更近。
  `toy_position_seed21_review_v1`；**不宣称两seed稳定完整身份恢复，不追加seed追正结果**。

### 3. 实际Attention与信息边界

- `retimed_attention_teacher509_v1/related_recent`完整latent/RGB等于旧recent；9记录FP32全pass。
- `retimed_actual_attention_v1`：first-denoise L0只source时间K变化，Q/全部V/其他K逐位相同；
  每记录source V/空间K不变、时间K等于delta64。max_abs0.018239、relativeL2最大0.001835。
- last-denoise L14 source mass9.676%→19.718%，L29 1.735%→4.542%；clean L29反而略降。
  只是32几何query/头×24heads，不是全Q/语义质量；clean不是第五次去噪，capture时间不可作方法速度。
- 官方顺序：首步L0 self-attn在当前文本cross-attn之前，固定noise/t/位置时Q0不含当前文本。
  `QUERY_SEMANTIC_AVAILABILITY_20260909.md`与官方forward AST单元测试（明确test doubles）。
- StreamingLLM/InfLLM已有位置重映射，已只读核对/锁SHA，见`MEMORY_POSITION_PRIOR_ART_20260909.md`；不拿rebase本身声称创新。

### 4. 同ROI实测与区域预算

- `native_region_layout_replay_v1`18点；v2补固定ROI缓存并去掉exact无用GPU gather，19点每点5warmup/30随机交错测量，全部K/V逐次精确。
- v2：固定ROI预打包0.3747ms/8.417MB/1copy；spatial4预pin0.8887ms/17.302MB/66copy；
  exact预pin2.1829ms/206copy；Block64预pin1.909ms，Frame预pin3.555ms。
- 单层真实K/V，不是视频加速/overlap。固定ROI创建7.09ms是单次观察；空间archive构建+pin成本单列，有波动，不给确定摊销阈值。
- `native_region_layout_report_v1`图与完整表；requested pinned形状预算2GiB，不是实测allocator reserved。
- `rebound_source_budget_v1`45次离线mask评估，full参考门禁全过。
  红区/随机非红/近邻非红/Venergy-matched均685同frame配额；Venergy-quarter1760为另一预算档。
  last L14 mean误差红0.0572 vs随机0.1205/近邻0.1245；lower-center红0.014 vs0.156/0.155。
  首L0误差仍约0.11；只离线局部误差，不是视频质量或已实现ROI在线方法。

### 5. 简单因果scene baseline已经真实跑通

- `native_condition_probe_v1`11文本/9构造请求：普通cosine选错settled初始空罐，最高相似度0.05内取最新9/9。
  不称9个视频，不称通用实体匹配。
- `native_causal_scene_memory.py`自动闭合scene最后8帧KV归档，8GiB FIFO；纯`SceneDescriptor` selector仅当前cue/T5＋过去描述符。
  floor0.8、margin0.05、minGap32，选完才读bank；当前原始文字是明确新增因果输入，禁止未来prompt字典。
  无source/target帧传入selector。只适用显式分镜返回，不是新颖router/正式25%稀疏方案。
- `causal_scene_gate64_v1`pass，完整actual latent/decodedRGB等于指定源recent gate。
- `causal_scene509_v1`5/5完整回归（bead13/settled19/settled20/toy13/toy21）都与对应指定源版本整段latent/RGB精确。
  `causal_scene_full_audit_v1.json`5pass/0missing；包括toy21负面输出。**没有新增5个独立质量样本**。
- 完整archive D2H7.7856768GB、CPU归档tensor峰值7.785725952GB、history H2D2.5952256GB；比指定源多归档3×，不隐藏额外成本。
  DiT约80.76–82.75s，单次计时不是速度门禁；selector CPU约0.4ms级，摘要/归档/安装全计入generation。

### 6. 新蓝画布首轮（不要当干净holdout）

- `blue_canvas_screen509_v1`4/4技术pass，seeds23/24×revisit/visible；pre64/pixel0:253配对。
- 长hold后source56–63仍有滚筒活动、图案变动，未通过干净停止/稳定性前提。
- 返回与可见cut控制均改成其他多色色块；不能全归检索。`blue_canvas_screen_review_v1/INTERPRETATION.md`。
- 当前正在跑的positive-stop只改第一句，既有负面结果不覆盖，selector文件SHA保持冻结。

## 后续工作取向

先闭合当前4条措辞控制与实际prefix审计，再决定新状态类别是否可推广。不要同时在新case上调selector。
区域局部误差提示语义分组有用，但早层/全视频与mask生产成本尚未解决；不直接写成“稀疏ROI已更快更好”。
当前仍无统一同5B骨干的新SCOPE/AdaCluster/SVOO排名，也无完整论文方法全面胜出。
官方native streaming VAE源码已初读：`streaming_decode = streaming_vae and not return_latents`；当前runner return_latents=True所以仍batch decode。
两GPU解码/传输流水是后续候选，**尚未迁移实现**，不能把仅翻config当已启用。导师旧1.3B系统收益/trace不改写。

`execution_inventory_v7.json`已关闭98个视频执行：95技术pass、3fail，missing0；包括gate/诊断重复。
当前4条positive-stop未入此快照。microbench、文本表征不混入视频数。
原H20 de72255没有重投或取消，本大轮没有刷新实时状态，不能把旧pending当实时核实。
