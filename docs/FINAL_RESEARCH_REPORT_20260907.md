# LongLive 方法—系统协同：研究结果与边界

本轮收敛为**无损执行优化＋跨层次 characterization**，不是已经成立的新 admission、三角色因果 memory 或自适应 KVOut 方法。正式方法集在生成前冻结；后来补齐的开发集消融没有参与正式选参。

## 1. 已验证的收益

三项配置：通用系统优化后的 RAG Dense、修正基础正确性的 legacy Final、同一 Final＋exact compact / archive-run pack / per-chunk roped union cache。三者均使用两 CPU 线程、pooled pageable archive 和相同基础正确性修复。新 utility、hierarchical raw cache、batched FA2、oracle/causal Tether 均未混入正式配置。

| 正式集 | 视频 / 独立 prompt-seed 组 | Final 系统版相对 legacy Final 完整耗时加速 | Final 系统版相对优化后 RAG Dense |
|---|---:|---:|---:|
| 477 pixel / 120 latent | 24 / 8 | 几何均值 **1.353×**，范围 1.129–1.652× | 几何均值 1.231× |
| 957 pixel / 240 latent | 12 / 4 | 几何均值 **1.430×**，范围 1.389–1.449× | 几何均值 1.337× |

以上实际硬件均报告 NVIDIA H200。同一组在同一卡、同一已加载模型内完成，Final 两系统的顺序交错。不是独立重复计时试验，不报告虚假的置信区间；477 与957的并行度、seed不同，不把两批绝对时间直接当作长度缩放实验。

完整耗时包含生成、VAE、检查、latent/RGB处理、编码与结果审计；模型加载单独报告，也提供三配置均摊加载后的数值。外层计时发现部分编码/落盘/元数据尾部波动达几十秒：这些仍保留在完整耗时中，不全部归因于 KV 优化。生产调用仍启用运行统计，不是无任何计时指令的裸 kernel。

全部 **12/12** 正式系统配对的初始噪声、有序 route、所有 latent 数值和编码前 RGB 完全一致；系统版继承 legacy Final 的质量，不是相对 Dense 的无损结果。历史 KV 的实际 H2D payload 减少 **80%**。

| 口径 | RAG Dense＋通用优化 | legacy Final | Final＋系统优化 |
|---|---:|---:|---:|
| 逻辑 history pair density | 100% | 25% | 25% |
| 累计 history H2D / 逐调用候选字节 | 20% | 25% | 5% |
| 477 全局 plan-executed pair density | 100% | 65.956% | 65.956% |
| 957 全局 plan-executed pair density | 100% | 64.141% | 64.141% |

25% 不等于全模型只算25%，5%也不等于只选择5%。GPU cache 在五次调用（四次去噪＋一次 clean commit）中实测80%命中：477为4080 hits/1020 misses，957为8880/2220。FA2内部 MMA tile padding、HBM transaction 不由这些逻辑计数推断。

## 2. 质量结论比单个分数更窄

正式36条视频对应24条不同 latent 轨迹；两 Final 系统共享语义审查。完成全部96份季度板、24份 overview 和必要的全分辨率细节检查。另完成6条 candle stress、4条独立轨迹、16份季度板；stress始终排除在正式均值、Pareto和选择外。评审是非盲 AI 视觉审计，不是人工偏好研究或逐帧播放认证。

- 狐狸：长时仍能保持可识别主体和雪林，但姿态、路径及雪花逐渐偏离 Dense。保真度差异不直接等于绝对质量排序。
- 宇航员：服装、头盔与温室保持，但面罩遮脸，不能据此证明“老人面部身份一致”；触摸/检查植物也未明确完成。
- Glassblower：场景、手和工具较稳定；反复拉伸/压平的发光材料并未清楚形成持续不可逆加工进展。Dense也有该限制。
- 蓝画布：477的seed20260908及独立957的seed20260910均出现 Final 后期凸起条带，Dense相对平涂。**粗颜色覆盖保留，不代表材质几何保留。** 这是原 Final 的可复现观察，不是系统优化造成的变化，也尚未证明具体的 KV 因果机制。
- Candle 两新seed：两种方法均未清楚展示缩短及蜡积累，支持将其作为困难负面stress而非有效状态代表。

PSNR/SSIM/LPIPS使用锁定 BF16 VAE 从保存的latent重解码、uint8 RGB转float32/255后计算；提供逐帧及各季度/late-quarter数值。PSNR延续相同帧100 dB上限，均值会受相同早期前缀影响。可视化中的 `state_retention=stable_sampled` 对蓝画布只指粗蓝色覆盖，材质/几何失败另列，不可扩展成完整状态成功。

发现并保留两类评估勘误：早期 canonical v1 把uint8传给要求[0,1]的指标，两个smoke分数已作废并在v2重算；初版拼图错误拉伸缩略图，未用于评分，v2逐像素重排并有单元测试。正式评分均使用修正流程。MP4编码差异已定位在相同原始VAE/RGB输出之后，但具体编码根因未被证明；MP4 SHA用于完整性而非模型等价。

## 3. 方法与系统的正、负证据

| 方向 | 结果 / 决策 |
|---|---|
| 连续索引 pack / per-chunk union cache | 通过真实视频无损与完整计费门禁；也给Dense同样的通用优化机会 |
| shared-union route compiler | 独立开发重复实验约6%完整耗时下降、输出和route精确一致；不等于新admission |
| 跨chunk raw/hierarchical cache | warm组件大幅改善，但完整视频仍mixed；Dense小全局raw slab会thrash，未推广 |
| batched FA2 | 完整backend下降42–55%，8视频精确等价；state Final完整耗时回退3.44%，按门禁不推广 |
| 成本感知 admission | held-out MAPE37.395% >15%；runtime禁用，不用正式结果回填成本模型 |
| utility / coverage /5% age-prior探索 | 原10-case校准9 pass/1基础设施fail，终态missing0；补充12个完整capture、72个精确字节匹配对照，仍无双类别worst-case不退化候选 |
| denoising / layer reuse | 自然shadow路线不等于强制cache相等；early/late双Plan做过完整teacher重放。相邻四层Jaccard低于0.8门禁，不跨层复用KV |
| 直接 `q_to_next_proto` | 真正源层Q＋目标层当时已存在的原型：召回26.33%/29.44%，补齐后的字节1.731×/1.700×；不推广，未捏造timeliness或视频收益 |
| D2H / page streaming overlap | 组件Nsight证明真实重叠；H200代表页流中H2D活动0.906ms、与kernel重叠0.160ms、父区间12.173ms。重叠存在不等于端到端改善 |
| Tether oracle | 完整Dense＋SAM2及两个时间索引版本实际跑通；是全候选传输的offline teacher，不进入在线Pareto |
| causal /三角色memory | 自动motion初始化与Q-role预测门禁未通过；state与identity可以重叠，不硬作互斥分区；无可靠因果角色视频推广 |

新补的因果探针必须同时保持 reference/capture 的完整latent、route和后续archive prototype一致，两类别均通过。初始未选fine block在随后两chunk重入teacher top25%的观测下界为8.58%–15.33%，三处超过10%；粗检索缺席的block另列为未观测。该结果支持研究反馈截断风险，不能把未观测当作安全删除，也不能直接给UCB/bandit正面结论。

## 4. 执行方式、真实瓶颈与流式上界

H200完整72点BF16/FP32-reference网格通过；RTX4090与RTX PRO5000 72GB Blackwell各12个冻结边界点通过。保留另外72点运行时报告为H800的结果，绝不按容量将其改名H200。每点5 warmups、30测量、median/p95，首调用/编译隔离。

当前Q-stationary家族在各scope均占至少90%的winner，不触发adaptive QOut/KVOut视频。表中的`grouped_fa2`是预准备FA2 reference，不能当作生产wrapper；KVOut是实际GPU reference，不是调优后的最佳实现。不同硬件只比较各自归一化结果，5kpro还使用了私有Triton3.3.1，硬件和工具链并非完全可分离。

RTX4090真实39/120/240 latent profile：Final稳态chunk p50约4.61–4.64s，Dense约6.19–6.26s；39只有两个稳态样本。last-chunk首调用里，Final Attention kernel约占父区间3.19–3.42%，完整backend约11.91–12.67%；二者是嵌套范围，不能相加。GPU idle不等于CPU算力饱和证明，但这些证据不支持把优化kernel作为主要视频收益来源。

CPU archive KV从120 latent的31,054,233,600 bytes增至240 latent的65,558,937,600 bytes，每增加一个latent增加287,539,200 bytes。**有界pinned/GPU缓存不是有界总历史存储，也不是已经可部署的无限实时流式系统。** 增量VAE prefix实验通过，但没有把它包装成完整在线服务。

Ncu在本地及H200都被驱动性能计数器权限拒绝。标准工具路径检查、GPU replay、失败日志均保留；未修改主机权限或共享环境。HBM/L2/片上transaction仍是明确的未获得证据，不以理论字节数冒充实测。

## 5. 论文归因与下一步

- A不成立：新admission未独立通过门禁，不能由cache收益冒充完整 Causal Reuse-Aware Paging。
- B未成立：没有双状态类别、等字节语义删除及可靠因果角色的证据支持三角色生命周期方法。
- C不触发：没有满足门禁的QOut/KVOut视频级自适应方案。
- 本轮可支撑：**无损历史执行优化＋因果信息、物理字节、复用轴、proxy-to-video失配及长期材质失败的characterization。** 未做新的LongLive2对照，不声称全面超过它或所有baseline。

后续最值得重新预注册的实验：在新的、Dense确实完成且可见的状态prompt上，分离颜色/覆盖与材质几何保持；用等字节干预验证边界/高频prototype或已提交状态版本是否有因果价值；联合粗frame检索与fine admission；在误差、排序与选择损失上重新校准成本模型；研究有证据约束的CPU archive保留/释放。它们是下一轮假设，不是本轮已证明的机制，也不应使用已消费的正式集调参。

## 6. 事实源

- [42视频与SHA索引](../../results/metrics/final_video_index_v1/index.html)
- [477配对表与图](../../results/metrics/formal477_research_report_v1/REPORT.md)
- [957配对表与图](../../results/metrics/formal957_research_report_v1/REPORT.md)
- [三硬件网格](../../results/metrics/dataflow_three_hardware_v1/summary.json)
- [长度/存储/作用域图](../../results/metrics/profile_scaling_report_v1/summary.json)
- [补充因果预取与预算消融](../../results/metrics/remaining_contracts_report_v2/REPORT.md)
- [总终态审计](../../results/metrics/program_terminal_audit_v1.json)

视频、capture和大trace属于相邻`results/`，不在Git仓库内；旧44/102-case与旧结果保持只读，没有训练。基线为`59020c0610cb48ffdf0ce36c32aba97378aa0b72`，正式生成源码为`58aa65bc0ad008ba0d767b4ff1ef1fbc0c24f859`，评价/分析源码与所有输入SHA在各审计中单独记录。
