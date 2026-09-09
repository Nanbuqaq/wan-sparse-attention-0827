# 当前探索断点（用户已解除审阅等待）

**最新精简入口：`docs/ACTIVE_HANDOFF_20260909_0958UTC.md`。** 它覆盖下方历史运行状态。
目前正在运行蓝画布positive-stop四条Dense控制，sessions94420/70547，勿重启。

## 最新覆盖项（先读，2026-09-09早间）

### 08:20 UTC实时覆盖：因果短gate已通过，5条完整回归运行中

最新已冻结运行代码`bd3d930002d7d83d50668736e23a39120a83737e`，全量634pass/1skip。
`causal_scene_gate64_v1/bead`已经技术pass并经`causal_scene_gate_audit_v1.json`核验：
自动archive ends8/16/48，选择8–15在48安装，整段actual latent/decoded RGB与指定源recent gate精确相同。
archive D2H1.132GB、history H2D0.377GB，原型D2H147456bytes；无人工source fallback。

**当前GPU0/GPU1仅有新批次**`causal_scene509_v1`5条完整回归，不能重启：
- 外层脚本`run_causal_scene_full_20260909.sh`；冻结worktree`/tmp/longlive-causal-scene.4gad99/checkout`@bd3d930。
- GPU0 / session99647：bead_fill_s13 → settled_s19 → toy_s21。
- GPU1 / session19810：settled_s20 → toy_s13。
- 两lane已dry-run并带物理锁启动，所有case预先冻结，包含已知身份negative。
- 完成后用`audit_causal_scene_full.py --root .../causal_scene509_v1 --references-root .../memory_activation_20260909
  --output .../metrics/memory_activation_20260909/causal_scene_full_audit_v1.json`验收。
  5条是方法运行等价回归，不是5个新的独立质量样本；完整archive3×2.595GB必须计入，不藏额外成本。

toy第二seed21已看完`toy_position_seed21_review_v1`：两种位置都改写肩部/身体，recent只是部分脸更近；
不能宣称两seed稳定完整身份恢复。不继续追加身份seed追正结果。状态类别与身份构造分开。

`native_condition_probe_v1/INTERPRETATION.md`已闭合；`native_region_layout_report_v1`为最终19点解释。
接下来的独立质量探索应优先新状态类别Dense-only可行性筛选，再冻结验证，不在现有正样本上继续调参数。
发布权限未恢复，不push、不新投InferHub；原H20状态本轮未refresh，保持不重投、不取消。

### 最新覆盖（07:50 UTC之后）：位置证据与真实搬运闭合，因果baseline准备GPU gate

以下覆盖所有较早“运行中”描述；CPU文件仍有待提交的新因果模块，不能reset/clean。

- `settled_memory509_v1`4/4完整视频pass，四份runtime审计均通过；
  `settled_memory_review_v1`关闭描述性审查。两seed recent恢复红珠，original/Dense空罐。
  `settled_memory_return_pages_v1`两recent全部128返回帧已看，没有原先大幅新倒入/搅动，
  但形状/填充外观/玻璃边界缺陷仍在，source也并非严格静止。条件性记忆恢复，不是完整方法胜出。
- `retimed_attention_teacher509_v1/related_recent`已pass，完整latent/RGB等于旧recent控制。
  `retimed_actual_attention_v1`9组新FP32回放pass；first-denoise L0除source时间K外所有输入
  逐位相同，各记录source V/空间K都不变。last-denoise L14 source mass9.676%→19.718%；
  clean L29反而略降。仅32个几何query/头、24heads，不是语义评分或方法计时。
- `native_region_layout_replay_v1`18点、`...v2`19点均通过逐次K/V精确比对；v1不覆盖。
  v2补固定ROI缓存并消除exact无用GPU gather。5预热/30随机交错重复：prepacked exact
  0.3747ms/8.417MB/1copy，spatial4预pin0.8887ms/17.302MB/66copy，exact预pin
  2.1829ms/206copy，Block64预pin1.9090ms，Frame预pin3.5549ms。
  创建成本单独且只有单次观察，不能给确定摊销阈值；不是完整视频或overlap收益。
  图表解释`native_region_layout_report_v1`。固定ROI能复用时，不应该强制大page。
- toy seed13四格`key_position_toy13_v1`已闭合；`key_position_toy13_review_v1`已审。
  related-recent显著更接近源脸/拼块/身体；late仍有人手进入，错误源仍花朵污染。
  单seed身份信号，不称完整跨主体验证。
- toy seed21 `toy_position_seed21_v1`两条related-original/recent都已pass（session52691已可回收），
  `toy_position_seed21_review_v1`刚生成/待看，**不得预判结果**；没有重跑错误源。
- `native_condition_probe_v1`已完成11文本/9构造请求。普通cosine选错settled初始空罐，
  最高相似度0.05内取最新在这9个请求均正确；只是文本角色表征，不是视频质量。
  `QUERY_SEMANTIC_AVAILABILITY_20260909.md`及官方forward AST测试确认固定noise/t/位置时
  首denoise L0 self-Q尚未接收当前文本；后续层/步骤不同。

**当前没有需要保留的旧GPU生成进程，接手仍应live核对。** 下一步新因果baseline：
`native_scene_admission.py`纯descriptor selector，`native_causal_scene_memory.py`自动closed-scene
最后8帧归档，8GiB FIFO；仅当前返回cue＋当前T5摘要，cos floor0.8、margin0.05、minGap32。
不传source/target帧给selector，选中后才读bank并复用recent绑定。是结构化分镜的简单基线，
不是通用实体跟踪或新颖router，也不声称正式25%稀疏预算。
runner新增`--causal-scene-memory`，与手工episode模式互斥；当前文本callback只按当前调用frame取值。
CPU已测纯规则、所有权、FIFO、一次安装，最后全量结果看工具/`causal_scene_CPU_pre_gate_v1.log`。
需本地commit冻结后，在GPU0跑64-latent gate（`--gate --episode-gate-layout --native-local-frames32`
及CFG1 positive/固定adaLN16/1），与`key_rephase_gate64_v1/raw_reveal/summary.json`做完整输出eq。
用`audit_causal_scene_gate.py`独立验收；期望archives ends8/16/48，自动选source8–15在48安装，
delta48，archive D2H3×377487360，history H2D377487360。这些期望只进入离线审计，不进selector。
**新因果GPU gate还没有启动**，不要把CPU实现当结果或直接扩视频。

最后一次全量此前627pass/1skip；因果模块新增后有632和随后634级回归，须以最后工具日志核对。
发布权限仍未恢复：不push，不新投InferHub；原H20未refresh、不重投、不取消。

### 实时覆盖：2026-09-09 06:17 UTC / 14:17 CST

**时间勘误**：本续进早先“13:40 UTC”等标题引用了服务器`date`/日志展示时间，其实际
时区为CST(UTC+8)。实验目录日期/顺序不变；明确UTC时间应读`date -u`或clock工具。
以下是真正当前状态，覆盖下面历史运行记录。

- 最新本地HEAD `81c6b543a3a123a40af148aed69d06b6605566a6`；最近全量617pass/1skip。
  后增1个Attention输入见证测试，定向已pass，尚未计入下一次全量。
- 位置组件长视频**2/2已pass**并关闭4格描述性视审，`key_components_review_v1`。
  phase_only delta16仍空罐；age_only delta48少量变形橙红内容，不恢复原红珠堆；
  完整recent delta64恢复类别但仍新动作。单seed组件，不是“禁用shot RoPE即可”的结论。
- **当前GPU0**：`settled_memory509_v1/seed20260919/original`原生完整runtime门禁；
  session69054，冻结`/tmp/longlive-settled-memory.Ftkcil/checkout`@81c6b54。
  外层`run_settled_memory_20260909.sh runtime run`，已dry-run且物理锁启动。不能重启。
  完成后运行`audit_settled_memory_runtime.py --case <上面case> --control
  .../settled_state_screen509_v1/lane0/settled_bead_revisit --output
  .../metrics/memory_activation_20260909/settled_memory_runtime_audit_v1.json`。
  通过后同脚本`lane0 run`（seed19-recent）和`lane1 run`（seed20-original/recent）各一次；
  后三条dry-run已经冻结。不要把runtime原生视频另算一条独立质量样本。
- **当前GPU1**：`retimed_attention_teacher509_v1/related_recent`，session61336；
  冻结`/tmp/longlive-key-components.HVKzJo/checkout`@bb13368。9条实际Q/K/V/O capture，
  必须与旧`key_position509_v1/related_recent/summary.json`整段noise/latent/RGB精确相同。
  外层`run_retimed_attention_capture_20260909.sh`，不能重启；先待它释放GPU1再启动settled lane1。
- 新CPU分析脚本`analyze_retimed_attention.py`，比较旧
  `attention_teacher509_v2/shot`与新capture，复用`attention_role_analysis_v1`。
  校验first-denoise L0除source时间K外所有输入不变；之后Q/currentKV允许随轨迹改变。
  所有source的V/空间K应不变，source时间K应等于声明delta64；9组新FP32门禁必须通过。
  只几何query样本，不是质量分、全Q均值或在线teacher入口；capture时延不得作方法计时。

新源码只本地commit，无push；公开源码只读访问StreamingLLM/InfLLM并已锁SHA，非外传研究数据。
原H20实时状态本续进未刷新，不能把历史pending当已核对；未重投或取消。

### 最新续进：新seed复验闭合，组件长视频运行中

`key_position_seed21_v1`四格已全部pass并完成描述性审查。源有效、pre96 latent及前381
decodedRGB实际精确匹配；H2D同为2.595GB。related-recent恢复红珠，original空罐；
两away都未恢复，且有草地侵入。晚段仍有多余动作，整体质量未成功。
`key_position_seed21_review_v1`含原图、季度、technical和semantic；seed13新版source图
在`key_position_review_v2`。同prompt双seed位置读取信号复现，不是正式holdout。

补充`key_position_color_proxy_v1`：使用既有HSV阈值，返回红色面积/自身source中位数
约0.906/0.724；original两seed中位均0。该指标也会命中手/瓶子，不是珠数、质量或物理评分。

**当前GPU0/1唯一新工作**：`key_components509_v1/{phase_only,age_only}`，seed13各一条；
sessions62258/67478，冻结worktree`/tmp/longlive-key-components.HVKzJo/checkout`@bb13368。
两个short gate都pass，CPU/GPU delta0/8/16/24/32/40/48/64数值通过；
`key_components_gate_audit_v1.json`确认actual pre48/source/phase/字节正确。
完整component source delta16/48，复用旧original/recent作为另外两格；不重复启动。
结束后`review_key_position_study.py --components --root .../key_components509_v1`
`--control .../context509_local_v1/lane1/shot --recent-control .../key_position509_v1/related_recent`
生成新`key_components_review_v1`。

主工作树正在准备`settled_state_v1`后续4条（19/20×original/recent），尚未运行。
新增明确协议开关与named-away source边界，避免五阶段误用segments[2]=32而取错源。
配置冻结了先前screen审查SHA；首seed19-original先做真实原生runtime门禁，对旧Dense实际
pre96和RGBprefix后才开后三条。不要把这个准备计为已完成GPU实验。

已只读核对StreamingLLM与InfLLM公开源码并锁SHA：位置重映射不新，见
`MEMORY_POSITION_PRIOR_ART_20260909.md`。网络仅下载公开文本，无push、无研究数据发布。
600pass/1skip是上轮全量；新增settled/color测试后的全量正在跑，稍后查工具57633结果。

### 13:40 UTC更新：位置四格闭合，新seed固定复验正在运行

完整CPU回归592pass/1skip（关闭无关自动pytest插件；默认自动插件在sandbox绑定socket失败，
不是代码测试失败）。之后补了review source坐标测试，需在最终回归再计数。

`key_position509_v1`全部完成，3新视频＋复用related-original，共4格technical pass。
实际pre96 latent/前381 decodedRGB精确一致，H2D相同2.5952256GB。related-recent恢复红珠，
但晚段额外倒液体；away-recent持续花朵/草地污染。不是整体质量胜出。
已在`key_position_review_v1/INTERPRETATION.md`和`semantic_verdicts.json`关闭描述性视审。
v1的source板是共同红珠参考；新review脚本已区分reference_state和实际selected_source，
避免把away格的参考图误认成选中的源。原v1结果不覆盖。

当前本地GPU唯一任务：`key_position_seed21_v1`，新seed20260921四格复验，
GPU0 related-original/recent，GPU1 away-original/recent；工具sessions74112/58380。
外层脚本`run_key_position_replication_20260909.sh`；同旧冻结worktree
`/tmp/longlive-key-rephase.vniDh5/checkout`@fb1afb0，不改变已过gate的运行代码。
冻结判读在`docs/KEY_POSITION_REPLICATION_20260909.md`；不可重复启动、选seed或结果后改delta。
两lane已dry-run并通过物理锁启动；代码仍不push，新InferHub不提交。

下一步先live核对四条summary；完成后review脚本用该组related_original作control，
生成新`key_position_seed21_review_v1`，验收全部prefix、来源、预算和新seed实际源内容。
若不复现，报告轨迹依赖而非自动增加seed追正结果。原H20状态未在此更新核查，勿当实时pending。

### 最新：cut组件分解已得到实质信号，历史K重绑定gate运行中

本地HEAD `fb1afb0c5af2b1a7ac382dbc0e8e052675144725`，全量591pass/1skip。
**发布权限仍未恢复**：最后确认外部push是7b18592；21ae000起的push被权限审查拒绝，
已请求用户确认目的仓库但无新授权。之后只本地commit/实验，不得换通道发布或新投InferHub。

已关闭的新Dense控制：

- `continuation_screen509_v1`四条，通过pre48/noise旧控制比对。两seed的nocut-anaphora
  和nocut-explicit都保留红珠类别，但物理漂浮/溢出/倒入仍在。不能把物理错误全算成记忆失败。
- `cut_components509_v1`六条＋复用native两控制，完整pre48 latent和前189 decodedRGB一致。
  `strip_words`单独去掉送T5的切换词未阻止seed19清空/seed20减量；`freeze_rope`保持source
  相位8则保留明显更多原状态，两seed一致。raw cut检测、pin事件和文字（freeze-only）都保持。
  `strip_words_freeze_rope`也保留红珠但仍有物理伪影。解释在`cut_components_review_v1/INTERPRETATION.md`。
  结论限定于可见状态控制，不是“所有镜头都应禁用Narrative RoPE”的泛化结论。
- 已读原生源码：cut prefix确实送入T5；RoPE与cut检测只看prefix，不把普通prompt变化当shot；
  cross-attention flags每chunk失效。相位跳变会改变Q对旧K的相对角度，但同一current块Q/K
  共享相位，所以它们内部相对关系保持。历史权重不是单调变化，不能只看source质量挑策略。
- `archived_key_rephase_v1`：固定已捕获Q/V、仅转source K的CPU诊断，phase-only delta16
  并非总增加source质量；recent-virtual delta64有不同局部作用，尚无视频质量结论。
- `state_region_diagnostic_v1`：已完成source红色连通token685/7040=9.73%，9条FP32数值门禁通过。
  比等token非红控制有更大局部输出敏感度，但颜色区域会包含相连落珠、未匹配空间距离，
  不能当语义重要性/视频质量。固定ROI的computed Block64覆盖51.8%，spatial4×4含padding20%；
  这些是计算字节，不是实测pack/H2D速度，不要写成已实现空间布局优化。

**当前GPU0/1唯一新工作**：`key_rephase_gate64_v1/{raw_reveal,raw_away}`，
脚本工作根`scripts/run_key_rephase_gate_20260909.sh`，sessions79323/77041；
worktree`/tmp/longlive-key-rephase.vniDh5/checkout`@fb1afb0。先live看summary，勿重复启动。
新算子只旋转存储K前44个时间通道，空间通道/V不变，真实source帧号保留，effective位置绑定
到target前8个virtual frames与current phase；这是新算法/新KV版本，不是物理layout等价。
`key_rephase_GPU_numeric_gate_v1.json`已过5个delta的CPU/GPU bitwise K对照、空间保持、FA2/FP32门禁。
短视频gate须验证pre48对旧bead gate完全相同、delta48/24、元数据不变和正确H2D；不看低分辨率质量。
通过后才考虑原生509的相关/away × original/recent位置对照，原相关shot控制可复用，其他case不能假设已跑。

统一入口`discussion_index_v1/index.html`已更新位置约定线索。原H20仍pending不重投。
本断点以下均为历史阶段记录，有冲突以本节为准。

### 当前最高优先状态：发布权限与新Dense筛选

**不要继续git push或借其他工具间接发布。** 推送21ae000时权限审查拒绝：
需要用户明确确认能向`https://github.com/Nanbuqaq/wan-sparse-attention-0827`外传研究代码/文档。
已在commentary向用户请求确认，尚未收到回复；没有绕过。最后确认成功push为7b18592。
21ae000及此后的本地工作未发布。新InferHub提交按“已push SHA”的手册前提暂停；
旧H20不取消、不重投。继续本地实验已明确告知用户并获本地GPU工具许可。

本地HEAD `21ae000b019bee0f29409b610815b420784d45c0`，全量580pass/1skip。
**当前GPU0/GPU1**：`settled_state_screen509_v1/lane0|lane1`，各两条Dense原生509，
source状态先停止/静止，再离开返回 vs始终可见但相同cut/RoPE边界的控制；seeds19/20
（完整20260919/20260920）。四case只做新工作负载有效性筛选，未运行新memory方法。
脚本`/home/zhouhe08/MyProjects/0904-longlive-system/scripts/run_settled_state_screen_20260909.sh`；
tool sessions64708/14533。worktree`/tmp/longlive-settled-screen.jFjmZW/checkout`。
不重复启动；先live核对。五阶段0/16/32/48/96，hold在32不加scene prefix，
实际cut indices2/6/12；native代码已核实只按prefix识别shot，cross-attention每chunk失效。
新protocol配置信息在`configs/system/native_settled_state_screen.json`，parser阻止screen期间memory干预。

隔离重编码两条gate和两条完整509都已结束，**4/4技术通过**。完整past/current两个版本
的pre96一致、buffer指针保持、一次重编码，GPU峰值仍24.375GB。持久source输入6.897664MB，
含当前条件hash临时量的CPU张量峰值11.091968MB；H2D past6.897664MB/current2.703360MB，
另有条件hash D2H4.194304MB和源重复D2D2.595GB。重编码约0.66–0.67s，完整计费。
但原条件变棕色颗粒、当前条件变杂色内容，晚段仍有新动作，尚无质量推广；
**不是原KV的无损压缩**。数据`semantic_remat509_v1`，审计`semantic_remat509_audit_v1.json`，
视审`semantic_remat_review_v1`。不用重跑这两条。

下一步：关闭以上重编码视审文字、审核新settled-screen的源段40–47是否真的停止，
以及visible-control在不离开时是否也自行改状态/出现动作；然后再决定新机制，不扩未验证任务。
统一入口已生成：`results/metrics/memory_activation_20260909/discussion_index_v1/index.html`，12链接验证通过。

### 更新：初始锚点与寿命已关闭，重编码gate运行中

- 源码已push `7b1859277192b8aeb76da564fadb13e1c0ffc376`，最新完整577pass/1skip。
- `initial_anchor509_local_v1`四条已完成并审查：source_only相关/错误均未恢复红珠；
  source_repeat相关恢复红珠，错误源不恢复，但相关晚段仍移动/散落。技术4/4，
  实际pre96和解码前381帧与旧控制相同。不是整体质量推广。
- `source_pin_lifetime509_v1`两条已结束：相关与错误都严格pre112 latent/前445 decodedRGB
  等于natural-repeat控制；此后才分叉，H2D/D2D/所有QK shape相同，只改480bytes pin元数据。
  保持第二份源未消除额外动作，不能把全部动作起因归到第三返回chunk时源副本被驱逐。
  对照及解释在`source_pin_review_v1/`，原始晚段在`initial_anchor_review_v1/`。
- `native_capacity_5kpro_review_v1`已检查所有8视频最后away128帧和返回季度、必要原图：
  均确有目标缺席。128窗口有toy独特脸/材质细节收益，但身体改写；两state仍无红珠。
  解释和semantic_verdicts.json已落盘，不能说大容量完全无用，也不能说足以解决状态。
- `execution_inventory_v1.json`关闭本轮已执行47项（44pass/3fail，含gate和诊断重复），
  missing0；不等于47个独立科学样本，原未启动H20另列。新重编码gate不在该快照内。
- phase4是clean KV写回，不是第五次去噪；它不修正已保存的当前chunk latent。
  18个Attention诊断的clean质量不能当当前生成已经利用记忆的证据。

**当前仅两张本地卡的新工作**：`semantic_remat_gate64_v1/{past,current}`，
由`scripts/run_semantic_remat_gate_20260909.sh 0/1`启动，工具session8151/64649。
冻结worktree`/tmp/longlive-semantic-remat.bssN5P/checkout`@7b18592。
它用已完成source8 latent（不是原KV），在空历史中分别配原条件/当前条件做一次t0 forward；
旧active context已被策略失效，所以复用同一套GPU KV buffers，不另分配完整KV。
之后恢复current clock/原source RoPE、复制源为两份、pin源。新KV是新storage version，
不是原KV精确replay。原1.3B pre/post recache负结果不修改。

接着先查2gate是否终态（不要重启已跑）。需验证：pre48与旧bead gate control相同、
buffer pointers不变、exactly1重编码、clock恢复、source2×、正确pin/shape、完整字节。
可用只读旧control：`results/videos/sprint24h_20260907/longlive2_native_placement_gate64_v1/lane1/none`，
seed20260904、[1,64,48,16,32]、pre48 SHA
`9f457f736e90628141f4d295d37f9309de0ba372b2013684c521c076817b3301`。
只有新gate通过才扩两个原生509的past/current重编码对照；其构造/条件区别要分开解释。

未提交的自己文件需保留/完成：build_native_attention_insight.py、collect_initial_anchor_study.py、
review_initial_anchor_study.py、audit_source_pin_lifetime.py、inventory_memory_activation.py及本断点。
用户原有0400/0500检查点和audit_retrieval_divergence.py仍不动。H20仍不重投；接手先live核对。

- 代码已push至`fa5062100857f0a15f55f2f5e1d7e23730bf3b29`，最新完整回归
  573pass/1skip。`scripts/build_native_attention_insight.py`是随后新建、待提交的图表工具。
- 本地`context509_local_v1` **8/8通过**，完整pre96 latent和解码前381张原分辨率RGB
  全部配对相同。完整视审结论：无稳定整体赢家；toy的正确源可恢复部分脸/材质，
  但身体改写和晚段零件仍在；bead reset变棕色碎粒并新增倒入，不是原红珠。
  入口`results/metrics/memory_activation_20260909/context509_local_review_v1/`，含
  `INTERPRETATION.md`、`semantic_verdicts.json`、`decoded_prefix_audit.json`和季度/原图。
- teacher v1两次在export之后报告文件大小时把st_size当函数调用而失败；PT数据保留，
  没有最终视频/latent验收，不可当合格capture。`34635e1`修复并让视频/latent先落盘。
  v2两次均整段noise/latent/RGB精确通过，各9条实际Q/K/V/O，查询32/头×24heads。
  `attention_teacher509_v2/`是事实源，v1失败不改写。两次均已结束。
- CPU `attention_role_analysis_v1`与`attention_role_figure_v1`：18条FP32回放均通过。
  max_abs0.0178466、relativeL2最大0.00182657、FP64标量归约的cosine误差最大1.67e-6。
  reset末次去噪L14 initial25.1%/source11.8%/current63.1%，L29 current96.9%；
  只是固定几何query样本，不是全Q均值或语义因果证明。v1微小负cosine来自FP32标量归约，
  新图表numeric_audit用FP64归约，原v1不覆盖。
- Blackwell独立容量批次已完成并回收：
  `zhouhe08__longlive2_native_capacity_5kpro_Iter0__f9b039679e15`，**16/16通过**
  （8短同compiler分配gate＋8完整容量case），新full seeds20260915/16，gate17/18。
  Source `f9b0396`，私有Triton3.3.1＋TORCHDYNAMO_DISABLE=1，未修改公共env。
  4组同卡32/128窗口noise一致、own-source轨迹可不同。KV10.381/41.524GB，
  峰值24.395/56.575GB；DiT约55–57/93–94秒，VAE约83–85秒。
  本地`results/videos/memory_activation_20260909/native_capacity_5kpro_v1/`，
  全文件SHA校验日志在`results/infrastructure/inferhub/native_capacity_f9b0396_v1/`。
  `native_capacity_5kpro_review_v1/`已生成；初步对照中两seed红珠仍丢失，
  toy仅部分特征回来/结构改写。还需完整away与季度审查关闭该批语义状态。
- 跨3.2/3.3精确失败始终保留。额外native128/3.3视频完成，与旧positive128/3.3
  的实际完整latent/RGB匹配，证明的是新的同compiler分配属性，不是重标旧fail。
  审计`same_compiler_allocation_gate_v1.json`、新原生case在`same_compiler_gate64_v1/`。
- 原H20 `de72255`仍pending（H节点失联），不重投，不取消；Blackwell不是它的重试。

### 当前仅有的本地GPU工作

`initial_anchor509_local_v1`四条原生bead/seed20260913视频，冻结`fa50621`：

- GPU0：source_only × raw_reveal/raw_away。
- GPU1：source_repeat × raw_away/raw_reveal。
- 使用工作根`scripts/run_initial_anchor_full_20260909.sh 0/1 run`启动，物理锁持有。
  工具session72030/46135，接手需live查文件/进程，不能再次启动。
- 临时工作树`/tmp/longlive-anchor-full.uinCOE/checkout`，用户旧文件不动。
- 两条新policy的短GPU gate已通过（`initial_anchor_gate64_v1`），且真实pre48
  与positive32控制完全相同。4-case dry-run保存于infrastructure/local/initial_anchor_fa50621_v1。
- source_only退出初始global，只留源8；source_repeat在global/shot重复同源8，
  显式逻辑multiplicity2（相当于source log2先验），不是padding。
  H2D均2,595,225,600 bytes；repeat另有等量D2D，only为0。
  预计full首/次返回K分别16/24或24/32，随后32，每次150 Attention调用；
  时钟global_end仍84480，only安装后local_end7040/pin-1/len0，repeat14080/7040/7040。
- 下一步完成4-case技术/pre96/shape/字节审计，与已完成lane1的none/shot/reset保持原始
  配对控制（不重跑），检查最新状态是否真的恢复、是否新增动作。单seed机制实验，
  未验证下一次切换与自动源选择，不能直接称在线方法成功。

### 没有实施，不要当进度或结论

source区域HSV/mask分解、KV低秩bias算子、VAE历史交换、inplace日志重建、
CPU逐层全历史onload、跨硬件portable noise bank均只是后续候选想法，尚未写/跑。
后续先按上面真实结果判断，不继续堆矩阵。全新讨论目标仍是因果长视频更快更好。

---

本轮开始于2026-09-08 19:49 UTC，文件日期按北京时间2026-09-09。
导师审阅包已交付并只读保留；用户明确让继续论文探索，无新的固定截止时间。
不主动spawn agent。两张4090可用，InferHub按当前手册走官方CLI；不改平台/公共环境。

## 已执行

- `08ee597`：新增global单chunk召回后恢复、原生128窗口、独立五臂运行器。
  5/5真实4090门禁通过，缺失0。global/TTL pre-return及首返回逐位相同，之后分叉。
  gate只证明技术，低分辨率目标缺席无效的旧结论不变。
- `de72255cbf26b696f7f1d3c22c8c2f38db33266e`：五臂收集/审查与InferHub入口。
  526 tests通过/1skip，已push。唯一20条原生509批次：
  `zhouhe08__longlive2_memory_placement_lifetime_capacity_Iter0__de72255cbf26`。
  实验为两开发任务×seeds13/14×none/global/global_one_chunk/shot/window128，
  完整seed为20260913/20260914。最后状态已claimed，H200节点CPUprep中，不能重投。
  后续提交的本地代码不改变这批冻结SHA。更新：20:48UTC前H节点心跳中断，
  平台自动回收到pending，requeue_count=1；没有生成视频/新提交。
- 源码推导的首返回上下文通过旧实际GPU pin位置核对：global有8帧reveal、16帧away、
  8帧当前；shot为8initial、8reveal、8away、8当前；window128则24initial、24reveal、
  48away、8当前。不是实测softmax质量或逐层KV值证明。
- `2821d9d`：CFG1不分配无用negative自注意力KV，所有正条件schema与官方AST函数一致。
  32与128窗口GPU门禁输出全部latent/RGB逐位一致；KV从3.020/12.080GB减半，
  生成峰值13.940→12.430GB、23.149→17.109GB。这是通用强基线，不是新算法贡献；
  还没有完整原生509容量/等价测试，不能把128全分辨率fits当实测。
- 私有Triton3.3.1在`TORCHDYNAMO_DISABLE=1`下解决旧CPU import AttrsDescriptor失败，
  真实4090生成也执行到video，但跨3.2完整精确等价门禁失败：noise相同，
  latent relativeL2=0.9211，第一chunk已0.2070。不得当小舍入误差或无损兼容；
  不据此直接重投Blackwell长批次。旧失败和新失败都保留。
- `f70bfae9d685f58801e567420007381d9e6c0c7d`：独立scene reset技术probe。
  源码frame组成及旧花朵污染提供动机：保留global8+召回source8，将away suffix
  在首返回denoise前通过native endpoint排除。不是zero K/V，不是纯layout，
  物理容量32不变但首返回执行K为24帧。相关/错误同字节源作为控制；仍是privileged admission。
  返回shape observer只读Tensor.shape，计数真实FA2的Q/K长度。CPU6项通过。
  最新全量为541pass/1skip（含shape observer）。两个context GPU门禁均pass：
  实际pre48与positive32控制完全相同，Q1024/K3072×150call，下一chunk K4096×150。
  新的`new_gates_audit_v1.json`关闭容量2pass/context2pass/编译器1fail，missing0。

## 结果与后台

工作根`/home/zhouhe08/MyProjects/0904-longlive-system`。

- 门禁：`results/videos/memory_activation_20260909/gate64_v1/`，terminal audit 5/5。
- 容量：`.../capacity_gate64_v1/`，positive32/positive128 pass，positive128_triton331 fail。
- 新context gate：`.../context_gate64_v1/`，由外层
  `scripts/run_scene_context_gate_20260909.sh 0/1`分别运行raw_reveal/raw_away。
  两条已完成，不要重复启动。正在准备新的本地8-case原生分辨率批次；见
  `SCENE_CONTEXT_LOCAL_STUDY_20260909.md`。这个批次不修改/重投H20。
- H20来源：`/kaimm-distill/zhouhe08/longlive-system/outputs/memory_activation_de72255_v1`；
  官方日志在`/kaimm-distill/infer_hub/queues/default/logs/<job_id>.log`。
  本地提交记录：`results/infrastructure/inferhub/memory_activation_de72255_v1/`。
- CPU frame分析：`results/metrics/memory_activation_20260909/frame_lineage_v1/`。
- CPU回归：`results/metrics/native_memory_study_cpu_20260909_v1..v4.log`。
- 冻结worktree：`/tmp/longlive-memory-activation.bTmP5W/checkout`、
  `/tmp/longlive-native-capacity.76Qv6t/checkout`、`/tmp/longlive-scene-context.xbJBu6/checkout`。
  源submodule从主checkout只读引用，不能编辑运行中的worktree/权重。

## 接着做

1. 监控并回收H20，不重复提交。使用`collect_native_memory_study.py`审计；
   `review_native_memory_study.py`生成四组季度、全away128帧、源/首返回/晚段和原分辨率关键帧。
   若某卡window128 OOM，保留硬件容量失败，其他四臂继续；不要误判质量失败。
2. context GPU门禁完成后核对source、pre48实际latent、原global/absolute时钟、
   同recall字节、metadata32→16、first-return实际K=24帧（每phase30层），后续K=32。
   不能用低分辨率质量推广。
3. 只有明确复核新实验问题与硬件runtime后，再冻结下一批；原H20不动。
   可考虑高分辨率reset_reveal vs等字节reset_away来测“记忆取回+旧场景退出”，
   给所有比较方CFG1公共容量优化。全窗口是强基线，不预设它会输。
4. 私有Triton跨版本漂移需单call/小kernel诊断或新的同版本配对控制；不能改公共env，
   不能用已完成H200视频作新Blackwell的精确/质量同轨对照。当前无Blackwell新任务。

旧三份未跟踪用户文件仍保留：0400/0500检查点和audit_retrieval_divergence.py。
导师新提交到de72255已push；之后本地新提交尚未push（接手需live核对）。
