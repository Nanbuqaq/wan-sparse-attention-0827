# 导师三项要求：当前结论与论文启示

这是研究中的证据汇总，不代表已经得到全面优于LongLive/LongLive2的新算法。
本次追加更新包含957系统重复实验、LongLive2原生参考与16条H200记忆视频。

## 1. 全系统分析：有分阶段证据，不能简单叫memory-bound

完整设备/存储/流程图见
[全流程拆解](SYSTEM_FULL_FLOW_AND_REPRODUCTION_AUDIT_20260907.md)。其中旧的
“尚未运行”状态已被后续结果推进，当前事实以本文件及事实源为准。

| 观察 | 量化证据 | 可以得出的结论 |
|---|---|---|
| 冷启动主要不是Attention | Dense477诊断：load296.57s，该阶段GPU活动union仅.318s；Final477中4次torch.load103.11s，load_state_dict36.29s | 存储、反序列化、构造/复制需要单列；不是GPU型号决定的Attention慢 |
| 原完整生成明显受主机执行组织影响 | Dense477诊断：生成host范围273.16s，本进程GPU活动81.59s，其余191.57s无本进程GPU活动 | 主机控制/同步/物化值得优化；不能据此断言CPU DRAM带宽饱和 |
| 非kernel Attention开销大 | Final477诊断：history.materialize_route38.48s，其中未被子范围覆盖22.49s；完整grouped backend39.58s，group KV整理17.78s | KV cache命中之外，描述符、索引、重排和dispatch仍可昂贵。嵌套范围不相加 |
| VAE是不同阶段的GPU工作 | Dense477诊断：VAE范围27.35s、GPU活动26.26s | VAE可与生成的空隙和其他工作重叠，但不能只凭活动比例判HBM/compute饱和 |
| 新流式路径确有同时执行 | Final153代表性Nsys：生成/控制kernel17.90s、VAE13.53s，两类kernel同时活动6.81s；parent41.65s | 真实时间线证明GPU工作重叠，不只是调用了async API |
| 显式复制不等于HBM总流量 | 同类153诊断，direct-outputRoPE令生成显式D2D从1.440TB降至.549TB | 数据整理冗余可很大；没有测到全部HBM/L2/片上transactions |

这些诊断有插桩和顺序效应，不能拼接成无插桩速度表。Ncu硬件计数器受权限
限制，失败记录保留且未更改驱动策略。因此目前应描述为**阶段相关的混合瓶颈，
生成中存在大量主机暴露空隙，VAE偏GPU活动主导**，而不是全程memory-bound。

真正速度依据是无profile的完整工作流重复实验：H200477的streaming×RoPE
48条执行均noise/routes/latent/RGB精确一致；组合配对中位数1.16–1.20×，保留
negative重复。directRoPE在async下3/4组中位数反而negative，不能一律开启。
首个MP4包从百秒级提前到数秒级；这是服务端mux时间，不是客户端显示时间。

## 2. KVOut/QOut：既测计算顺序，也测真正不同的搬运次数

用户提出的方向已经分成两类实验，不能混称一件事：

| 控制对象 | 实验与结果 | 边界 |
|---|---|---|
| 同一共享union，KV各H2D一次 | Q-stationary/KV-stationary实GPUreference；H20072点、4090/5kpro各12边界点 | 隔离GPU计算/读取顺序；不能声称降低PCIe字节。已有网格不支持adaptive视频推广 |
| 同一逻辑group→tile图，有限驻留 | 高共享cap4：BF16Q-major288copies/108MiB，KV-major96copies/36MiB；完整replay102.29→76.49ms | 重复是容量与请求顺序自然产生；非完整视频、原预制page布局准备不在该旧replay内 |
| 真正CPU pack→有限buffer→H2D→partial Attention | 新frame-major生产者含pack/slot等待；Q4680 eager consumer34.96→34.57ms，几乎无收益；小page配置producer更慢 | 独立线程不保证关键路径变短 |
| 降低消费端发射成本后再流水 | 相同Q4680 CUDA-graph consumer：serial16.82ms、same-thread async13.63ms、producer13.65ms | graph和pipeline分别归因；并非与最强eager-union或完整视频的胜负 |
| Nsight验证供给重叠 | bounded graph representative：H2D2.783ms，与kernel重叠2.652ms，parent13.884ms | 真正overlap已测到；仍需完整模型RoPE/route/准备成本，不能直接推广视频 |

系统实现覆盖连续run、persistent pinned、严格per-chunk cache、route metadata复用、
GPU group描述符驻留、有限容量调度、CPU producer、partial softmax合并、CUDA graph
consumer及完整视频VAE流水。生产history H2D/D2H overlap仍未启用；FA3/FA4专用
TMA/WGMMA pipeline没有实现。混淆这些层级会夸大成果。

核心motivation：**KV中心调度可以在有限显存下减少重复取数，但只有供给和消费
都提前组织好，这些字节收益才可能成为暴露等待/端到端收益。**

## 3. Tether：足够判断效果和启示，不继续追求完美复现

已使用官方CF+AE权重、无LoRA并保留显式runtime修正。两seed茶壶477均有
reference/Tether/neutral三臂；两seed骑车自动mask失败。手工oracle恢复仅一seed
成功，另一seed跟踪仍失败。完整结论见
[Tether长视频](TETHER_LONG_VIDEO_FINDINGS_20260908.md)。

Tether目前没有稳定整体质量优势：茶壶0出现额外手部/光圈，茶壶1后段失去
提示要求的居中主体。neutral零bias也出现明显长期轨迹/检索差异，不能把一切
变化归因于semantic bias。它没有产生在线速度结论，也没有和旧AdaCluster/
SVOO/SCOPE形成同骨干、同长度、同预算的新排名。

值得保留的是分组及其生命周期，而非two-pass形式：不同信息可拥有不同影响力、
存储表示和更新时间；有组织的组关系为稀疏执行创造条件。但当前实验也说明，
“保住某一颜色/区域”不等于正确保存身份、场景和状态，mask获得与后续对齐也
必须计成本、测可靠性。

## 距离论文主结果还缺什么

已有可信系统改善与多项排除性证据，尚无稳健的“算法质量＋系统速度”共同赢家。
两个最重要的未解问题是：可靠的因果记忆激活/信息绑定，以及有界长期存储。
本轮有界KV版本probe仅用于诊断，两seed没有稳定身份恢复赢家，不能当成方法完成。
LongLive2原生对照与独立分镜记忆测试已经完成；其骨干/分辨率不同，且源码本来已有多种系统优化，
不会把我们1.3B上的通用优化自动宣称为超越LongLive2的贡献。

新增957的12条完整执行、三次配对重复均保持route/latent/RGB一致：motion配对
中位1.1325×、state1.1552×。首MP4包约5.3–7.1s；仍非16fps实时生成，CPU历史
归档仍65.56GB。两条unique轨迹的季度视审发现主体漂移/状态非单调，系统保留
了原质量缺陷，而不是改善这些缺陷。

新LongLive2分镜任务真正让目标离开画面，返回后两seed身份改变、两seed红珠状态
丢失。相关旧KV可部分取回信息，但16条H200四臂实验的global-prefix替换引入
明显场景污染，没有稳定整体质量收益。raw/log同准入全视频逐位相同，CPU归档
张量约90×差异；独立进程记录数值配方后可精确重建KV。warm-CPU测试仍是raw
搬运更快，不能把“存得小”写成“恢复更快”。这给出冷日志/热KV的机制动机，
但自动准入、隔离场景污染和有界长历史策略尚待解决。

## 直接事实源

- `../../results/metrics/full_flow_20260907/`（使用REPORT_V2及修正预览）
- `../../results/metrics/bounded_schedule_20260907/`
- `../../results/metrics/sprint24h_20260907/system_insight_figures_v1/index.html`
- `../../results/videos/sprint24h_20260907/streaming_factorial477_h200_v1/factorial_audit.json`
- `../../results/metrics/sprint24h_20260907/streaming_priority_trace_v2.activity.json`
- `../../results/metrics/sprint24h_20260907/page_graph_node_trace_v2.activity.json`
- `../../results/metrics/sprint24h_20260907/episode_snapshot477_review_v1/INTERPRETATION.md`
