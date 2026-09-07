# 第二轮：全流程与搬运调度的直接证据

本轮不把 eager union、所有query共享同一选择、或25%物理传输预算当作不可变前提。逻辑去重账本、GPU驻留策略、Attention消费顺序是不同层次。

## 全流程拆解与量化

流程/设备/存储说明见 [完整拆解](SYSTEM_FULL_FLOW_AND_REPRODUCTION_AUDIT_20260907.md)。新增四个真实诊断：Final39/120/240 latent与Dense120；全部插桩/无插桩latent精确一致。

| Dense 120-latent阶段 | CPU范围wall | 实际GPU活动union | 解释 |
|---|---:|---:|---|
| 加载/初始化 | 296.57s | .318s | 本次主要是存储/CPU初始化，不是GPU算力打满 |
| 生成 | 273.16s | 81.59s | 约70%范围内没有该进程GPU活动；需研究主机控制/等待/物化 |
| VAE（含像素D2H） | 27.35s | 26.26s | GPU活动占主导；还不能进一步断言HBM或compute饱和 |
| CPU RGB转换 | 1.59s | 0 | 旧range名字误写D2H，实际像素已在CPU |
| 编码/落盘 | 2.78s | 0 | CPU阶段 |

这是instrumented诊断，不能与其他运行直接拼出生产速度。GPU活动通过CUDA launch correlation归因，包含memset；不把CPU范围与CUDA service相加。实际HBM/L2计数器仍未获得，管理员只读Ncu也返回ERR_NVGPUCTRPERM，未改驱动策略。

生成中的显式H2D约78.40GB，显式D2D memcpy约4.771TB。后者不包含所有kernel读写，绝不能叫“GPU全部HBM流量”。query-group整理内约6.717GB H2D主要来自路由/索引元数据；原KV去重不等于控制数据和GPU重排已无冗余。

新profile初版预览漏VAE归一化，已保留旧文件并从保存latent修复4/4；新预览RGB精确匹配canonical公式。原正式42视频未受影响。请用各目录`REPORT_V2.md`及`preview_normalized_v2/audit.json`，不要用旧预览判断质量。

## 实际改变H2D次数的新对照

同一逻辑group→KV tile图、同一KV缓存容量、同一partial backend；query-major的重复取数由容量与访问顺序自然产生，不人为添加重复copy。KV-major每块服务全部相关query后再释放，windowed在两者之间折中。eager union容量不够时明确标infeasible。

| 实验/共享程度 | Q-major copy / KV-major copy | 字节变化 | 完整replay median |
|---|---:|---:|---:|
| Q1560、FP32 partial、高共享、cap4 | 72 / 24 | 27→9MiB | 23.991→17.606ms，1.36× |
| Q4680、BF16 FA2 partial、高共享、cap4 | 288 / 96 | 108→36MiB | 102.291→76.494ms，1.337× |
| Q4680、中共享、cap4 | 192 / 96 | 72→36MiB | 67.686→55.678ms |
| Q4680、不共享、cap4 | 96 / 96 | 不变 | 35.763→35.213ms，接近 |

每个主网格5 warmup、30测量，各42实测配置；BF16 partial相对FP32 reference的示例relative L2约.0021。GPU KV backing容量固定；这不是GPU总显存，其他Q/softmax状态等另计。

边界：synthetic page-major fused K/V，原archive转成该布局的成本未计；不是实际视频收益，不是原FA1 kernel获胜。logical graph digest是group→tile图，跨shape还须匹配Q/head/dim/exact参数。FP32初版精确source snapshot已按manifest SHA补存。

代表性异步KV-major Nsight：H2D24次/9MiB/.6678ms，copy/kernel真实重叠**0**；parent27.013ms、GPUbusy2.748ms。开stream没有形成有效供给流水，需要研究前瞻式producer/批量发射，而不是继续把异步API当成功。

## 接下来最值得验证的三条机制

1. **更多有价值历史，配合流式执行**：放开全局25% union cap，保留per-query概率质量；在真实显存/延迟预算下比eager、windowed、KV-major，不只比字节。
2. **已知粗候选的跨层预取**：chunk开始已经知道各层粗frame IDs，可预取完整粗候选再在实际Q到达时细选，不必沿用已失败的跨层Q内积预测。测extra、ready、争用与净等待。
3. **真正的生产者流水与元数据驻留**：将CPU pack/copy调度和GPU消费解耦，缓存不变的GPU路由描述；与VAE分阶段瓶颈结合。近期排除窗口也可能给archive提交提供可用延迟余量，须保持逻辑archive时间不变并做正确性门禁。

这些是被观察支持的后续假设，不是已经成立的端到端优化。

## Tether及其他方法当前结论

Tether官方权重CF+AE、无LoRA已跑通81帧reference/mask/Tether/neutral控制；只在最后chunk使用历史，末段缩略图没有可信绝对质量赢家。新477冻结批次正在做两prompt双seed与neutral控制。骑车seed0的自动AMG选了公交车，跟踪48%而非95%，保留为失败；不能用这种mask评判骑车人identity。

旧AdaCluster两例4090 OOM、SVOO/ SCOPE旧迁移结果不能与这组跨骨干/长度/软件直接排名。新比较须统一backbone/retrieval/decoder/信息与实际预算，并保留强native Dense路径，不能为了统一接口让Dense承担不必要的稀疏调度开销。

## 事实源

- `../../results/metrics/full_flow_20260907/`
- `../../results/metrics/bounded_schedule_20260907/`
- `../../results/metrics/tether_source_gate21_review_v1/`
- `../../results/metrics/tether_official_vae_continuity_20260907.json`
- `../../results/videos/tether_runtime_fixed477_v2/`
