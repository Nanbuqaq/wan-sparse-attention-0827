# 全流程、系统技术与论文复现审计（第二轮）

本文件接续第一轮报告，不改写旧实验。目标是流式长视频的质量—完整延迟—资源 Pareto；shared union、25% H2D、现有 Final 都是对照点，不是新研究必须保留的限制。

## 1. 当前流水线从开始到结束

```text
磁盘/共享存储中的权重、配置、prompt
  → CPU 构建模型/读权重 → generator/VAE 上 GPU，text encoder 动态换入
  → 文本编码 → GPU 上 prompt embedding / cross-attention cache
  → 初始化 GPU local KV cache、CPU archive、pinned staging
  → 每个 3-latent chunk：
      历史 latent descriptor 粗帧检索（所有层共用候选 frame IDs）
      → 四次 denoising forward：
          输入 patch/time/text embedding
          → 30 个 transformer block：
              norm/QKV projection
              → 更新当前 KV、处理 GPU local window 与 CPU archive
              → Q summary / CPU history route
              → cache 查询 / CPU gather-pack / H2D / RoPE
              → history 与 exact/current/recent Attention
              → output projection / residual
              → text cross-attention / residual
              → FFN / residual
          → output head / flow-to-x0 / scheduler 加噪
      → denoised latent 提交至 CPU 输出，更新检索 descriptor
      → 一次 timestep=0 clean commit forward，提交历史 KV
  → latent 回 GPU、连续 VAE decode
  → RGB 转换、D2H → CPU 视频编码/落盘
  → 帧数、finite、SHA、latent/RGB 等价审计
```

| 阶段 | 主要设备与存储 | 做什么、如何影响效率 | 新的 profiling 范围 |
|---|---|---|---|
| Python/import/CUDA 启动 | CPU、磁盘、CUDA context | 冷启动、模块初始化；不混入稳态 chunk | import wall，单独列 |
| 模型构建/权重读取 | 共享盘→CPU RAM | Wan、T5、VAE、AR checkpoint；当前骨干还加载 LoRA | load pipeline、torch.load、load_state_dict |
| 模型 dtype/device 准备 | CPU→GPU | BF16 转换、模型驻留、text 动态换入安装 | module.to；与子范围嵌套 |
| 文本编码 | CPU tokenizer、GPU T5、动态权重驻留 | 文本一般每视频一次；不是每个 denoising step 重新编码 | text.encode_with_dynamic_swap |
| cache 初始化 | GPU exact/local KV 与 cross-attention cache | 容量/分配开销、初始零填充；CPU archive 起始为空 | 两类 initialize_cache |
| 粗帧检索 | descriptor 张量所在 GPU；结果少量回 CPU | 已提交历史 descriptor 与最近 descriptor 相似度 top-k，默认 6 帧；候选不同会影响所有层 | history.coarse_frame_retrieval |
| descriptor 更新 | 当前 GPU latent→小 descriptor | 当前实验 avg_pool；官方 Tether 用 AE，成本与检索语义不同 | history.descriptor_update |
| generator 输入/输出 | GPU | patch/text/time embedding、head、flow-to-x0 | embedding/head/forward 与未归因 self 范围 |
| self-attention 投影 | GPU 权重/Q/K/V | QKV linear、norm、output projection；不能把全部 self-attention 时间叫 kernel | self_attention.q/k/v/o/norm_* |
| archive offload/index | GPU→bounded pinned→CPU pageable；CPU prototypes | 旧 KV 出 local window，保存 raw KV 与路由原型；必须验证 DMA readiness | 现有 NVTX offload/index + 新 archive 范围 |
| 细粒度 history route | Q summary GPU→CPU；CPU prototypes/元数据 | 选择逻辑边。Final 当前有 70/15/15 预算与 shared union；不是所有候选的通用约束 | 现有 route NVTX/服务计时 |
| pack/pin/H2D | CPU archive→pinned staging→GPU | offset 排序、连续 run、padding、copy 数、缓存命中影响；更少 bytes 不保证更快 | materialize_complete 与 GPU activity |
| RoPE 与 query-group 整理 | GPU | raw history 位置变换；按 query group gather/cat 可能重复复制 KV | 现有完整 backend 与 group_kv_replication |
| Attention kernel | GPU HBM/L2/片上 | QK、softmax、AV；IO 与算力归因需硬件 counters | CUDA activity；Ncu 计数器目前未取得 |
| text cross-attention | GPU，文本 KV cache | 第一次建立文本 KV，后续复用；与 history self-attention 不同 | transformer.cross_attn |
| FFN/norm/residual | GPU | 非 Attention FLOPs、activation 读写、Python dispatch | transformer.ffn/norm；残差留 block self 范围 |
| scheduler | GPU | 相邻 denoising step 的重新加噪 | scheduler.add_noise |
| latent 提交/clean commit | GPU→CPU latent；再次 GPU forward | 四次去噪后还有一次 clean commit；不是四次总调用 | latent.commit_to_CPU_output；call_in_chunk=4 |
| VAE | latent CPU→GPU、GPU decode | 必须保持 temporal cache；影响 first frame、chunk latency 和显存 | vae.decode_complete |
| RGB 与编码 | GPU→CPU、CPU codec、存储 | uint8/布局转换、MP4 编码、落盘；不能算入 Attention 加速 | RGB_convert_D2H、MP4_encode_write |
| 结果审计 | CPU/GPU、磁盘 | 解码帧数、finite、SHA、保存 latent；研究 overhead 与用户服务分列 | validation、latent_save、decode_integrity_check |

本轮新增 `profile_full_flow.py` + `full_flow_profile.py`。插桩只修改当前进程的函数包装，不修改只读 upstream；先做 CPU 测试与 39-latent CUDA 无扰动回归，然后扩 120/240。它记录嵌套 wall、CUDA stream span 和层/chunk 标签；同 seed 关闭插桩的 latent 必须精确一致。

CUDA stream span 含 host 发射空隙，不是服务时间；CPU self time 是未被子范围覆盖的控制/dispatch/等待，不等于纯 CPU compute。主机范围、CUDA activity 与关键路径分别报告，不相加制造一份饼图。初版全流程实测仍在执行，不能用上一轮代表调用补成全视频数字。

第一轮已有证据：4090 Final 240-latent 代表调用 2.1768 s，GPU busy union 0.4347 s、Attention kernel 0.0745 s、完整 backend 0.2757 s。它支持继续研究主机/执行组织，不证明 CPU DRAM 或 GPU HBM 饱和。

## 2. 系统技术到底实现到哪一层

| 技术/论文思想 | 已有实现 | 真实视频状态 |
|---|---|---|
| 连续 run、archive-aware packing | `archive_pack.py`、`transfer_plan.py` | 已进入正式 Dense/Final 系统路径 |
| 预分配 pinned staging、K/V 分开或融合 pack | `staging.py`、layout replay | persistent separate 已进入；融合不预设胜出 |
| 去噪轴 per-chunk roped cache | `history_cache.py`，严格版本/位置 key | 已进入；实测五次调用 80% 命中 |
| raw/hierarchical cache | raw slab、重 RoPE、组合缓存 | 组件与视频做过，完整延迟 mixed，未推广 |
| batched FA2/减少 query-group wrapper 开销 | `grouped_staging.py` | 8 视频等价；有端到端回退，未推广 |
| GPU→CPU 双 ring / D2H-compute overlap | `offload.py`、`benchmark_offload_overlap.py` | replay/Nsight；生产 `offload_overlap` 仍被显式拒绝 |
| Page256 H2D-compute 双 buffer | `StreamingDataflow` | synthetic replay/Nsight；生产 `onload_overlap` 仍被显式拒绝 |
| verified prefetch：预测、actual 补 miss | `prefetch.py`、route probes | 计划/trace/replay；不具备正式集端到端收益 |
| previous_route / q_to_next_proto | 因果 trace 分析 | 后者约 26–29% recall、约 70% extra bytes，不推广这个预测器；不否定其他预取 |
| Q-stationary / KV-stationary | 实际 Triton GPU reference，后者 partial+merge | 72 点 H200 + 两硬件边界；不是生产 video backend |
| FA3/FA4 TMA/WGMMA 专用流水 | 阅读/设计参考 | 未实现，不能归到现有成果 |
| 分层/多 GPU stage pipeline、异步服务 | 设计假设 | 未实现 |
| 有界 CPU archive 释放/压缩 | 分析与测量 | 未实现；120/240 latent CPU KV 31.05/65.56 GB |

重要代码事实：`runtime.validate_runtime_system_config()` 对非 none overlap 抛出 NotImplementedError。调用异步 API、拥有双 buffer 类，不等于视频系统已启用或有净收益。

## 3. 不让 union 限制研究空间

需要分开两个问题：固定逻辑 Attention 边，比较系统执行；改变每个 query 的高价值历史，比较方法质量与系统联合 Pareto。前者能归因，后者允许改变 union、传输量、预算，只需如实记录。

预注册的最小执行比较不是只把旧 QOut 改名：

1. query-major：按 query-group 的逻辑边请求 KV，显式有限 residency/LRU；记录自然产生的复用/重复，不人工添加无用 copy。
2. eager shared-union：当前对照，先一次性物化 union 再执行。
3. windowed union：小批 query-group 共享一个工作窗口，折中工作集、首计算等待与重访。
4. KV-major page streaming：每块搬来服务所有相关 query；正确合并 online softmax，双 buffer 重叠。
5. hot resident + cold streaming：高重用块驻留、长尾流式；比较单位显存节省的真实时间。

有价值的新假设：粗帧检索在 chunk 开始已给出所有层候选 frame IDs，因此可以预取下一层**完整粗候选页**，不必预测下一层 Q/route；实际细路由到达后只消费逻辑集合。这允许额外字节换隐藏等待，但必须测带宽争用、extra bytes、timeliness、峰值显存及端到端。

另一假设：不再强制每个 query 的 union=25%，保留 per-query 概率质量，再用分页/窗口/重叠承受更大的 union。必须与相同完整延迟或显存预算下的 eager-union 比较，不能用不等预算直接宣布更好。先 capture/replay 验证，再视频；既有正式集不得重新充当未见测试集。

## 4. TetherMem：本轮发现的原配置复现缺口

官方源码已存在于项目 `.runtime/sources/tethermem`，源快照为 `f9ebf718995ad162aa5b32e96a34b36a01a6c2d7`；本轮 `git ls-remote` 核实官方 HEAD 仍为该提交。不是缺源码，而是之前没有按其全部配置执行。

| 维度 | 官方 `configs/tethermem.yaml` | 此前 153 帧 oracle 迁移 |
|---|---|---|
| AR generator | `causal_forcing.pt` | `longlive_init.pt` + LoRA |
| retrieval descriptor | `ae_latent_mem.pt` | avg_pool |
| 入口 | 官方 `scripts/run_tethermem_pipeline.py` | 本项目 sparse runtime adapter |
| mask | 官方首帧自动 subject + full-video SAM2 | 已完成 9 帧前缀自动初始化 + oracle 全视频传播 |
| 时间地址 | 官方原行为 | source-compatible 与 latent-aligned 两个变体 |
| 验证范围 | 尚待本轮执行 | 单 prompt/seed，39 latent/153 pixel |

因此旧结果必须标为 **TetherMem 机制迁移**，不是官方配置效果复现。官方 `target_avg=.25` 是 attention prior 权重约束，不是 25% KV 物理传输预算。

本轮先补原配置：复用已有 Wan/SAM2 文件，下载官方 pinned HF revision `aaafe325726e0204ada9d1f019356fde97e08c2e` 中两个缺失推理权重（约 6.039 GB），禁止以 avg_pool/LoRA 替代。运行官方原入口，保留代码/权重/配置/视频 SHA、SAM2 overlay、两 pass 总成本；独立审查源兼容版本和时间对齐修正，不偷偷改变官方行为。

原配置跑通后，再在同一骨干、retrieval、prompt、seed、长度、VAE/评价下比较迁移方法；原论文复现与统一骨干对照分别成表。

## 5. 与 AdaCluster / SVOO / SCOPE 的现有证据

旧 44/102-case 目录只读，以下仅为历史事实，不与新 Tether 数字直接排名：

| 方法 | 已有事实 | 不能推出 |
|---|---|---|
| AdaCluster-AR | 两条 477 基础 case 在 4090 OOM；没有成功的同长度质量对照 | 原算法质量差；H200 也不可运行 |
| SVOO-AR | 两条 477 成功；旧主体/倒水 LPIPS 约 .369/.251；候选传输约 97.8%/95.7%，CPU route 很重 | 在修正系统/不同骨干下必然更慢；弱于 Tether |
| SCOPE-AR | 8 条 477 扩量均值 LPIPS .261，生成段 speedup .32×，传输约100%；有957结果 | Tether 优于/劣于 SCOPE；这是原论文完整复现 |
| Tether oracle 迁移 | 单状态153帧；两版本 raw-RGB LPIPS .048/.051；后期管子更稳但可能抑制动态 | 长期效果优于前三者；原配置复现成功 |

这些方法的时间、硬件、骨干、长度、数据和协议不统一。旧 pipeline 还有后来发现的计时/线程/readiness 问题，绝对时间只能按原实现解释。新的“谁更好”结论必须来自新的同条件对照。
