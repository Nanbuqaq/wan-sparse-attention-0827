# QOut、KVOut与FA4：把资源模型接到LongLive

这是导师系统分析的一部分，不启动新方法探索。配套量化报告和Perfetto阅读说明
会列出真实测量；本文区分论文事实、现成实现、我们已测的结果与待验证迁移。

## 1. 先把“搬运”拆开

```text
CPU DRAM archive
  ── PCIe/H2D ──> GPU显存（4090：GDDR6X；H200：HBM3e）
  ── GPU cache/load ──> L2 / shared memory
  ── MMA operand reads ──> Tensor Core
                            └─ 输出到register（Hopper）或TMEM（SM100 Blackwell）
  ── 非矩阵单元 ──> softmax、exp、归约、索引、rescale
```

CPU→GPU每次缺页搬入、不同CTA对GPU显存/L2的重复读取、MMA对shared operand
的读取是不同问题。FA论文主要处理GPU内部；外层KV onload必须另行计数和计时。
共享union只是一个执行选择，不限制我们研究有限驻留、分页和异步供给。

## 2. QOut/KVOut究竟改变什么

这里的Q指query tile/group，不是逐个query token。QOut/KVOut是本项目的数据流
简称，不能把某个Python外循环与整个FlashAttention库简单画等号。

```text
QOut / Q-stationary
parallel for Q tile i:
    Q_i、输出累加器和softmax统计由这个CTA负责
    for 它需要的KV tile j:
        读取K_j,V_j，更新这个Q tile的输出

KVOut / KV-stationary
for KV tile j:
    使K_j,V_j可用
    for 需要它的Q tile i:
        更新Q_i的输出与softmax统计
```

| 对象 | QOut倾向 | KVOut倾向 |
|---|---|---|
| 输出所有权 | 每CTA私有O/m/l，容易并行、少跨CTA归约 | 多个Q的状态需要驻留、反复写回或partial+merge |
| KV复用 | 多个Q tile可能重复请求同一KV；L2可吸收部分读取 | 同一KV供多个Q消费，但跨CTA共享不自动成立 |
| 状态空间 | 单Q tile的O和统计较小 | 全部Q的O状态太大时，节省KV读取会换来Q/O读写 |
| 并行度 | 沿Q维有大量CTA | 沿KV维切分可能要归约，或减少Q并行 |
| 固定逻辑边的计算量 | QK与PV的有效FLOPs相同 | 不自动减少QK/PV；可能改变归约、rescale和冗余工作 |

共同的有效Attention矩阵计算量是

\[
F_{attn}=2\sum_{b,h}\#\text{有效QK边}_{b,h}(d_k+d_v).
\]

LongLive的\(d_k=d_v=128\)，所以是每条边512 FLOPs；FMA按2 FLOPs计。
这不是实际MMA指令数：tile padding、mask和部分输出归约还会产生工作。

### 为什么用户提出的方向确实可以减少H2D

在**有限驻留**下，query-major执行可能反复淘汰/取回同一KV；KV-major先让
所有相关Q消费当前KV，再释放，能减少这些真实缺页。简单例子：Q0需A/B，
Q1需B/C，Q2需A/C；只能驻留一页时，按Q执行有5次缺页，而按KV执行只搬3页。
两者仍计算相同的6条组间关系，但KV-major要维护多个Q的部分输出。这个示意
是串行单页，不偷偷加入第二个prefetch buffer。

**若union已经一次性全驻留，或per-chunk cache命中，则两种顺序本来就可以
只H2D一次。** 此时不能继续把KVOut收益说成PCIe字节减少，应看GPU内部读取、
执行/归约开销和并行度。FA2作为内层kernel与外层KV-major供给并不矛盾。

### 实际做过的两种比较

| 比较 | 已测结果 | 能说明什么 |
|---|---|---|
| 同union各H2D一次的resident/onload数据流reference | H20072点、4090/5kpro各12边界点；Q-stationary家族在各scope≥90%winner | 该reference网格不支持adaptive视频；不是优化后KVOut的终极性能 |
| 同逻辑图、有限residency，高共享cap4 | BF16 Q-major288copies/108MiB，KV-major96copies/36MiB；replay约102.29→76.49ms | 实际减少重复H2D；这个旧replay的预制page准备不在时间内 |
| 包含CPU pack的Page256/cap4/Q4680供给 | eager consumer串行34.96ms、producer34.57ms | producer线程本身无明显净收益 |
| 降低consumer发射成本后 | CUDA-graph串行16.82ms、same-thread async13.63ms、producer13.65ms | graph与异步分别归因；额外线程没有独立收益 |
| 上项代表性Nsys | H2D2.783ms，其中2.652ms与kernel重叠；parent13.884ms | 有真实overlap，但不是完整视频速度，也未包含生产全部route/RoPE成本 |

最后两条有可导入Perfetto的真实trace。不能只看“调用了异步API”。

## 3. 当前LongLive实际是什么shape

1.3B路径每chunk为3 latent帧，每帧1560 tokens，总Q=4680，12 heads，head dim128。
稳态候选history通常6帧=9360 tokens，exact/current/recent=9360 tokens。
Dense总K=18720；Final的25%历史选择给出history2340，总K=11700。

Final的`query_block_size=64`用于Q摘要评分；**legacy shared-union执行把所有Q
消费同一union，实际记录是12个active query groups（每head一个），不是888个
64-token执行段。** grouped FA2把head当作varlen sequence打包，每段Q仍为4680。
应读取RoutePlan/执行记录，不能拿摘要粒度代替实际kernel形状。

当前chunk内部调用`causal=False`，因果性由历史/当前chunk的构造保证。这与
标准长序列下三角causal prefill的worktile不平衡不同，因此LPT不天然适用。

## 4. FA4真正值得学：先算“每种资源要多少cycle”

依据FA4 v1 §2.2、§3.1、§3.2，**B200/GB200的论文模型**取：

- BF16 MMA：8192 FLOPs / cycle / SM；Hopper参考是4096。
- MUFU指数：16 exp / cycle / SM；论文指出B300/GB300可为32，不能混用。
- shared operand读：128 bytes / cycle / SM，论文引用微基准。

对Q tile \(M\)、KV tile \(N\)、head dim \(d\)，forward两次MMA：

\[
T_{MMA}=4MNd/8192,\qquad T_{exp}=MN/16.
\]

该流水中QK是shared/shared，PV是tensor/shared。按论文128×128 MMA粒度，
shared读取字节为

\[
B_{smem}=2\lceil M/128\rceil\lceil N/128\rceil(256d)
 +2\lceil M/128\rceil\lceil d/128\rceil(128N),\quad
T_{smem}=B_{smem}/128.
\]

| 论文forward tile | MMA cycles | SMEM cycles | exp cycles | 结论 |
|---|---:|---:|---:|---|
| M=N=d=128 |1024|768|1024|MMA与exp相当，不能只优化矩阵乘 |
| M=256,N=d=128 |2048|1536|2048|更大tile仍需重叠softmax，不是自动更快 |

理想重叠的迭代下界接近各资源时间的最大值，而不是把它们全相加；真实依赖、
启动/收尾和资源竞争会增加时间。这里是**论文资源模型**，不是本机实测cycle。
式2转写时必须先数bytes再除bandwidth，结果与论文Table1核对，避免量纲混淆。

### Forward：具体改了什么

1. 每CTA处理两个128-row Q tiles，ping-pong让一块做softmax时另一块做MMA。
2. SM100 MMA输出到TMEM。两个softmax warpgroup、一个correction warpgroup、
   驱动MMA/TMA的warpgroup分工；把O rescale移出softmax关键路径。
3. 一行128个score会产生很高register需求；P分段写出以减小峰值活跃寄存器。
4. 只对约10–25%的exp使用FMA多项式模拟，其余走MUFU，平衡吞吐与寄存器/延迟。
5. 条件rescale保留一致的旧scale和最终归一化，减少不必要的输出向量缩放。

**精度不能省略。** 三次多项式的FP32相对误差高于硬件exp，但BF16舍入后误差
主要由量化决定；这不代表与原输出逐位一致，也不保证长视频轨迹不分岔。
公开softmax源码在`scale_log2`空间判断阈值8，对应因子256；不能把8照搬到
未缩放QK内积或自然对数空间。我们未把这些近似操作植入正式视频。

### Backward：为什么不能把它的收益写到LongLive inference

论文标准backward为5次MMA。128³例子MMA2560 cycles，SMEM3328，后者高约30%。
2-CTA通过分摊operand、DSMEM交换和重排归约降到SMEM2688 cycles，且减少dQ原子
归约。这是清晰的资源模型→设计→消融闭环，**但本任务不训练，没有这个backward**。

### LPT不是只把最长任务排前面

FA4还保留batch外层、head的L2局部性，限制head分组不冲掉L2，再反向遍历causal
Q blocks；varlen需要预处理排序与映射，可复用metadata。论文报告H200上的4–8%
MHA、7–14% MQA8收益。这是论文对应shape结果，不是我们的LongLive收益。
当前legacy Final每head共享同长度union、chunk内非三角mask；只有新route产生
明显不等长工作，且波次/尾部证据支持，才值得把LPT作为具体优化。

## 5. 网上已有算子：可复用到哪一层

| 来源/算子 | 可以拿来做什么 | 注意 |
|---|---|---|
| FA1 Algorithm1及历史CUDA实现 | KV外循环与online softmax的机制参考 | 不等于现代PyTorch可直接安装的最佳KVOut实现 |
| FA2 `flash_attn_func` / `flash_attn_varlen_func` | 本项目已使用的高性能QOut基础；可作为外层分页供给的consumer | 输入布局/原始K/V/RoPE/边集合固定；packed与wrapper成本另计 |
| FA3 Hopper实现 | H200上TMA/WGMMA、warp specialization的硬件路径参考 | 4090不支持同样指令，论文H100速度不直接外推 |
| FA4 CuTe-DSL | SM100 kernel、softmax/scheduler/block-sparse扩展基础；当前源码也有Hopper路径 | 发布期与当前commit区分；当前README包名`flash-attn-4`，发布期为`flash-attn4` |
| Triton fused-attention tutorial | 可读的Q-block并行online-softmax参考 | 教程不是最强baseline，也不是CPU archive系统 |
| FlashInfer paged prefill / attention state merge | 页式KV接口与partial输出合并，可用于外层KV-major原型 | LSE约定、布局、mask和dtype必须核对；split-KV不自动等于KV-stationary |

FlashInfer的merge-state文档给出float32 LSE及形状，但“logsumexp”这个名字不足以
证明不同库的对数底数/缩放约定一致；接入时还须核对实现及测试，不能直接拼接。
源码可用不代表已集成或有视频收益。

## 6. 对我们现有系统的优化优先级

| FA4式做法 | LongLive对应证据/动作 | 当前状态 |
|---|---|---|
| 分资源而非统一“memory-bound” | 区分GPU GEMM、GPU copy、CPU准备、PCIe、VAE和交付 | 本次量化表与Perfetto提供 |
| 先保证供给再追Tensor Core | 连续run、persistent pinned、cache、减少metadata与重复物化 | 多项已进入真实视频；CPU pack成本仍需计入 |
| 依赖图与真实重叠 | bounded page consumer graph；生成/VAE分流；ready与slot生命周期 | 有代表性真实重叠和独立完整视频证据，两层不混淆 |
| 分析非MMA开销 | 未融合LoRA、norm/RoPE/索引/调度；shape匹配GEMM校准 | 本次实际计数，不以1.3B名称粗算 |
| 精细kernel调度/数学近似 | 仅在kernel暴露占比与不均衡支持时考虑LPT、更多stage或softmax优化 | 不在用户过目前新开算法/质量矩阵 |

LoRA merge可减少额外矩阵乘，但BF16下“先合并权重”与原计算顺序不保证逐位
相同；不能直接宣称无损。CUDA graph/GPU子图、减少索引/变换往返也需完整计费，
不能用kernel局部降时替代视频收益。现成FA4不直接解决CPU route和PCIe等待。

## 来源与版本

- FA1：[2205.14135v2](https://arxiv.org/html/2205.14135v2)，历史`74af0233…`。
- FA2：[2307.08691v1](https://arxiv.org/html/2307.08691v1)，发布期`4f285b35…`。
- FA3：[2407.08608v2](https://arxiv.org/html/2407.08608v2)，beta`418d6771…`。
- FA4：[2603.05451v1](https://arxiv.org/html/2603.05451v1)，发布期
  [`a365a190…`](https://github.com/Dao-AILab/flash-attention/tree/a365a1909c081744693177255a23c669c0f208fa/flash_attn/cute)，
  本次在线核对current main仍为[`ce088ab9…`](https://github.com/Dao-AILab/flash-attention/tree/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute)。
- [Triton教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)、
  [FlashInfer attention](https://docs.flashinfer.ai/api/attention.html)、
  [merge-state](https://docs.flashinfer.ai/generated/flashinfer.cascade.merge_state.html)。
- 原文和算子文件SHA在`results/metrics/mentor_system_audit_20260908/sources_v1/`
  与`sources_operators_v1/source_lock.json`，没有以旧笔记代替原文读取。

来源细节：FA4主文写B200，而附录A.1写B100 180GB SXM6/1000W并出现“March2025”
与论文日期不一致；本文保留这一出处差异，不自行改成已核实硬件。论文1613TFLOP/s
及1.3×/2.7×仅按主文场景引用，且其caption说明更新版cuDNN已吸收部分优化。
不能将这些数字直接用于H200、4090或SM120的RTX Blackwell卡。
