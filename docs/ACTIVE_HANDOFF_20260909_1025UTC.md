# 继续点：2026-09-09 10:25 UTC

## 10:40 UTC最新覆盖：完整VAE gate已通过，当前无运行GPU任务

session54933已完成并回收；已live核对GPU0/1均空闲，未保留其他GPU工作。
`native_vae_full_gate_v1/gate.json`为pass：batch重放匹配旧RGB，chunk8/7/13全部509帧浮点和RGB逐位一致。
max_abs/relative_l2/1-cosine全0。batch peak allocated14,937,192,448，chunk8为12,148,815,872 bytes。
raw.float基准是原生BF16 decoder；完整解释在该目录INTERPRETATION.md。
**不要重跑这些gate**；下文“正在跑”是历史状态。

下一安全工作：实现并验证真正的生成→VAE worker→增量sink流水。可保留原DiT return_latents=True，
用clean-commit hook在最终chunk生成后提交owned latent，独立GPU的stateful VAE逐group输出。
须有物理双GPU锁、有界队列/pinned槽、明确ready事件和错误传播，完整latent/RGB门禁先于长profile。
不得直接调用原生缺失的cached_decode，或翻return_latents配置却失去latent验收。
每个物理GPU的峰值与服务时间分开记录，实际时间线证明overlap；总wall不能由重叠组件求和。
当前没有实现两GPU流水，更没有端到端速度结论，不能把这次解码内存下降当最终“更快更好”。

## 10:38 UTC覆盖更新：低分辨率通过，原生分辨率正在跑

低分辨率gate已pass：chunk8/7/13全部浮点像素逐位相同，RGB SHA也等于旧记录；旧session45560已回收。
**当前GPU0**：`run_native_vae_full_gate_20260909.sh run`，session **54933**，物理锁。
输出`results/metrics/memory_activation_20260909/native_vae_full_gate_v1/gate.json`，日志
`results/infrastructure/local/native_vae_full_gate_v1/run.log`。冻结执行代码00b5c7b，执行文件SHA已核对。
输入是已保存settled seed19/recent的128 latent、704×1280、509 pixels，比较native batch与chunk8/7/13。
新增GPU峰值allocated/reserved记录；reserved可能继承先前allocator缓存，不能把它当独立路径显存占用。
CPU误差/哈希检查计入本门禁wall，不用作速度结论。没有新DiT视频，不换VAE。
接手先检查此full gate，禁止重复启动；下文“当前小gate”是历史描述。

若full gate也通过：保持原生DiT return_latents=True，可用已验证clean-commit hook取得最终chunk latent，
送有界队列的VAE worker；明确GPU device、pinned槽、ready event和生命周期，不在传输完成前复用槽。
逐pixel group送sink，同时保留完整生成latent验收；必须用真实时间线证明供给/解码/输出重叠。
这些pipeline步骤尚未实现，不能因适配器通过而声称两GPUoverlap已成立。

上一goal turn有实际进展；goal仍active，不重定义目标、不误标完成。完整历史看0958UTC交接。
不push、不新投InferHub、不绕过发布授权；用户三个旧未跟踪文件不动。不spawn agent。

## 已结束的后台

`blue_positive_stop509_v1`4/4技术通过，旧sessions94420/70547已回收，不重启。
`blue_positive_stop_review_v1`核验actual pre32/前125RGB等于旧措辞，positive两分支pre64/前253RGB也相同。
两个source56–63全部32像素帧已看：seed23滚筒向左移动但仍在画面中，seed24继续在条带上动作。
返回和可见cut控制仍改成多色色块。单句改写不足以达到稳定source，保留negative，停止在同prompt上追正。
没有memory方法进入蓝画布；selector参数/文件SHA未修改。

## 当前GPU0门禁

- 脚本工作根`/scripts/run_native_vae_stream_gate_20260909.sh run`。
- tool session **45560**；GPU0物理锁。接手先核对是否已结束，不能重启。
- 输出`results/metrics/memory_activation_20260909/native_vae_stream_gate_v1/gate.json`，日志
  `results/infrastructure/local/native_vae_stream_gate_v1/run.log`。
- 冻结代码`8a475e3`，最新CPU **643pass/1skip**。运行直接使用publish_repo，两执行文件有SHA校验。
- 只加载原生VAE与已保存64-latent低分辨率输入，比较native batch和chunk8/7/13，逐像素浮点/原RGB SHA。
  不是新DiT视频，不是FP32模型比较，也不是两GPUoverlap性能测试。

## 为什么做它

已核对真正配置`vae_type=wan`→WanVAEWrapper→WanVAE_，没有cached_decode。
pipeline streaming路径却调用cached_decode；只有另一个LightVAE实现了此方法。不能暗换VAE/权重。
当前runner `return_latents=True`还关闭streaming，所以不能只翻配置宣称已实现流水。
新`native_vae_stream.py`保持原decoder逐latent、first_chunk、feat_cache行为，跨chunk不清缓存。
conv2是1×1×1；仍需GPU数值门禁，因为不同chunk形状可能改变CUDA运算结果。
每次yield只给新pixel group，不做全视频增长cat。提前放弃iterator会让流失效；reset才可新视频。
CPU测试还检查generator yield不泄漏torch inference mode到调用者。

## 下一步

1. 核对VAE gate.json与数值。若失败保留，不改门槛；区分baseline重放是否本身匹配旧RGB、stream分块误差。
2. 若通过，做原生704×1280保存latent的完整解码重放，再谈两GPU供给/解码/编码时间线与总成本。
3. 原生pipeline目前没有直接逐chunk客户端输出callback，且return_latents=False才进入streaming；
   后续须保留latent验收、线程/stream的真实同步，不能把已调用异步API等同于overlap。
4. 状态算法方向仍有未解决身份结构与动作/状态分离，不能用VAE系统结果代替算法结论。

新增本地文件：native_vae_stream.py、gate_native_vae_stream.py、test_native_vae_stream.py、NATIVE_VAE_STREAM_GATE_20260909.md。
已完成102个本轮命名视频执行的最新inventory将另存v8；这个数字不是旧102-case矩阵，也不是独立科学样本数。
