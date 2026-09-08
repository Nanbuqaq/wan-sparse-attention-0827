# 当前探索断点（用户已解除审阅等待）

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
