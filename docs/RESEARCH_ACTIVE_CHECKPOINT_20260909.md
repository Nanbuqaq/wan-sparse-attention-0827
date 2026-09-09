# 当前探索断点（用户已解除审阅等待）

## 最新覆盖项（先读，2026-09-09早间）

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
