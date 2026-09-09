# Native5B 双设备时间线：怎么看、哪些数不能相加

文件：工作根下
`results/metrics/memory_activation_20260909/native_pipeline_profile509_v1/audit/native_two_gpu.perfetto.json.gz`。
在 https://ui.perfetto.dev 的 **Open trace file** 选择本地文件即可；不需要分享或发布实验数据。
本次用已有官方Perfetto v58.2解析器实际导入成功，没有error/fatal统计；验证报告和SQL在
`results/metrics/memory_activation_20260910/native_perfetto_parser_validation_v1/`。

先看三条轨道：

- `CPU native pipeline scopes`：主机端生产、提交、等待和收尾的NVTX范围。
- `GPU 0 · generator`：DiT生成的真实kernel/copy事件，不是纯self-attention轨道。
- `GPU 1 · VAE/pixels`：原生VAE解码、像素整理和输出D2H。

搜索`native_pipeline/full_run`，观察整段98.359s，再选一段中间chunk放大。
上下两卡kernel条同时出现，才是本实验的跨设备计算重叠；图中的空白不自动等于CPU计算。
选中事件可读名称、开始/持续时间、device和copy的bytes。

| 可核对的量 | 当前这条诊断轨迹 |
| --- | ---: |
| GPU0 kernel服务时间 | 74.694s |
| GPU1 kernel服务时间 | 77.209s |
| 两卡同时执行kernel的区间并集 | 58.844s |
| 至少一张卡有kernel/copy/memset的时间 | 94.080s |
| GPU1像素D2H | 5.504GB / 0.219s |

两个kernel服务时间不能直接相加当作完成时间；CPU范围还包含嵌套和等待，其累计199.051s
也不是CPU忙了199s。两卡同时有活动不等于SM满载，更不是功耗测量。
本条中，每张卡自己的copy与kernel重叠为0；证明的是**两卡计算重叠**，不能冒称已证明
同卡H2D/Attention异步隐藏成功。H2D、D2D、kernel内部HBM访问须分开。

这是一条与原生完整latent/RGB等价的509帧诊断，不是独立质量样本，也不是多次生产测速。
此前172.735→98.866s仅为同双卡serial/overlap的一组门禁观察；初始化不在该计时窗口。
多GPU低延迟不自动意味着优于两条独立单卡任务的总吞吐。下一次系统扩量需要补这个等资源对照。
