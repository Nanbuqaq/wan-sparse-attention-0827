# Perfetto阅读说明：从全流程到某次Attention

## 1. 打开哪份文件

解压审阅包，在[ui.perfetto.dev](https://ui.perfetto.dev)点 **Open trace file**，
直接选择`.trace.json.gz`，无需先解压gzip。网页在本地浏览器解析文件；本任务没有
上传或分享任何trace。远程工作区的文件可先通过编辑器文件面板下载。

推荐按下表顺序，一次开一个文件：

| 文件 | 内容 | 先回答的问题 |
|---|---|---|
| `traces/final39_overview.trace.json.gz` | Final39从加载到输出的完整CPU阶段＋GPU活动曲线 | 时间主要花在哪个阶段？ |
| `traces/final39_detail.trace.json.gz` | 最后一个chunk首forward的所有CPU scope、CUDA API、GPU kernel/copy与关联线 | CPU在算、准备、发射，还是等GPU？ |
| `traces/current_streaming_overview.trace.json.gz` | 当前优化的生成/VAE流水，全41.654s范围 | GPU空隙与流水线状态如何变化？ |
| `traces/current_streaming_detail.trace.json.gz` | 上项20–26s窗口，含全部CUDA API | 两条GPU stream是否真的同时工作？ |
| `traces/page_overlap_detail.trace.json.gz` | 13.884ms的CPU pack/H2D/partial-Attention组件窗口 | H2D和kernel是否真实重叠？ |
| `traces/dense39_overview.trace.json.gz`、`dense39_detail.trace.json.gz` | 对应Dense39诊断 | Dense与Final各自的阶段构成，不当跨卡速度排名 |
| `traces/dense477_overview.trace.json.gz`、`dense477_detail.trace.json.gz` | 旧477完整诊断与late-forward窗口 | 长轨迹如何展开？注意其版本/日期与39不同 |

overview是便于浏览的缩略视图，**没有假造kernel条带**；其中GPU曲线来自真实
kernel/copy区间并集，按5ms计算。detail保留真实事件（边界处明确标记clipped）。
时间坐标沿用对应完整捕获；不要把detail视图的时间当作另一条独立运行。

![本次Final的实际阶段和GPU活动曲线](figures/final_timeline.png)

## 2. 三类轨道怎么读

### CPU · NVTX / CUDA launch

- `startup.load_pipeline`：模型构建、权重读入、dtype/device准备；展开可看到
  `startup.torch_load`、`startup.load_state_dict`、`startup.module_to`。
- `generation.complete`：包含文本编码、cache初始化、逐chunk生成。
- `generator.forward`：点开事件参数，看`current_start_token`、`call_in_chunk`
  与`phase`。本模型每chunk是4次denoising＋1次clean commit。
- `transformer.block`：点参数中的`layer`看第0–29层；再展开Q/K/V/O、FFN、
  cross-attention、route、物化等子阶段。
- `cudaEventSynchronize`、`cudaDeviceSynchronize`、`cudaStreamSynchronize`
  等是CPU等待API；它们可能没有产生新的GPU kernel，不能在导出时漏掉。

CPU彩条长度是函数/标记范围的wall时间。它可能包括Python工作、内存访问、
发射与等待，**不是纯CPU算术时间**。

### GPU · actual kernels and copies

- 每条track对应一个真实CUDA stream。
- `kernel`是设备执行；名称中`flash_fwd_kernel`可帮助定位Attention。
- `H2D`、`D2H`、`D2D`是实际copy事件。点开查看`bytes`、`corr`、`leaf`和`major`。
- 事件参数保留完整kernel名，CPU/GPU关联由CUDA correlation ID建立。
- 点击GPU事件并查看flow，可回到发起它的CPU API。GPU可能在CPU范围结束后
  才执行，不能只按横向时间重叠猜它属于哪个函数。

### GPU activity fractions · NOT SM utilization

`kernel active %`等counter表示一个5ms时间桶中，至少有对应活动的时间比例。
`any traced GPU activity %`是所有kernel/copy/memset区间的并集。

**100%不代表所有SM满载，也不代表Tensor Core利用率100%。** 本任务没有获得
Ncu的DRAM/L2/SMEM事务或动态occupancy计数器；报告中寄存器、shared和CTA上界
来自launch metadata，和这条曲线是不同指标。

## 3. 建议的十分钟路线

1. 打开Final39 overview，先找`startup.load_pipeline`，确认加载与生成分开。
2. 搜索`generation.complete`并聚焦。界面支持搜索、选区缩放；选中slice后可用
   `F`聚焦，W/S缩放、A/D平移（也可使用工具栏和鼠标）。
3. 同时看CPU generation范围和GPU活动曲线。CPU很长而GPU曲线有大片空隙，
   说明存在主机暴露空隙；尚不能判定CPU DRAM已经饱和。
4. 打开Final39 detail，找`generator.forward`。参数应指向latent36的首pass
   （`current_start_token=56160`，每latent1560 tokens）。
5. 找某个`transformer.block`，分别看`self_attention.q`、`transformer.ffn`和
   Attention core GPU kernel；不要把CPU函数返回时间作为GPU服务时间。
6. 查看物化/route附近的CUDA API，区分大量小copy/分配/同步与真正大payload。
7. 点击一个H2D事件，读bytes与duration，再沿flow找CPU发起点。H2D不全是KV；
   本模型首次T5编码还有约11.36GB参数换入。
8. 打开current streaming的overview/detail，两条GPU stream同时有kernel才是
   设备并发证据。仅CPU的“提交”条带重叠不能证明GPU重叠。
9. 打开page overlap，观察橙色H2D与蓝色kernel的横向交集。它证明组件流水，
   不证明生产history-onload已启用或完整视频收益。
10. 回主报告的阶段表，用同一次run的FLOPs和GPU服务时间计算有效TFLOP/s。

![真实页流水窗口](figures/page_overlap.png)

图为从原始时间戳重绘的引导图，不是伪造的Perfetto截图。原始trace仍是事实源。

## 4. 三个最容易误判的例子

| 看到的现象 | 正确解释 | 不能直接推出 |
|---|---|---|
| FFN CPU范围约1.39s，GPU服务约5.01s | 发射与执行异步；算TFLOP/s用GPU服务 | FFN超过GPU峰值、CPU时间就是全部计算时间 |
| FA2静态warp占用上界约16.7%，有效约142TFLOP/s | 少量warps也可能把Tensor Core服务做得不错 | occupancy低必然是主要瓶颈 |
| 当前两类kernel服务17.897s＋13.531s，有6.807s同时运行 | 需要看区间并集与关键路径 | 把组件时间直接相加当总时间，或随意减一个overlap数 |

层级关系是：self-attention模块包含投影、Rope/缓存、backend；backend又包含
整理与核心Attention。报告GPU列按leaf归因，CPU inclusive列仍含子范围；
例如self-attn行的CPU48.09s不是“除去Attention后的剩余时间”。

## 5. Perfetto SQL例子

在SQL Query区域运行。**不要对全部slice直接sum(dur)**：那会同时加CPU父子
范围、CUDA API与GPU工作。

```sql
-- 按真实GPU kernel名称聚合服务时间；仍不是端到端关键路径。
SELECT name, count(*) AS calls, sum(dur)/1e9 AS service_s
FROM slice
WHERE category = 'kernel'
GROUP BY name ORDER BY service_s DESC LIMIT 20;

-- 看CPU等待API；名字取决于CUDA/runtime版本。
SELECT name, count(*) AS calls, sum(dur)/1e9 AS host_wait_api_s
FROM slice
WHERE category = 'CUDA_API' AND name LIKE '%Synchronize%'
GROUP BY name ORDER BY host_wait_api_s DESC;

-- 传输事件参数可在界面直接点开；args表也可按arg_set_id查询。
SELECT id, name, dur/1e6 AS ms, arg_set_id
FROM slice WHERE category IN ('H2D','D2H','D2D')
ORDER BY dur DESC LIMIT 20;

-- 查看数据导入错误。正常交付文件应没有非零error。
SELECT name, severity, value FROM stats
WHERE severity = 'error' AND value > 0;
```

这些是slice统计，不是新的硬件counter。detail有裁剪边界，不能拿它的局部事件
总量当完整视频总量；报告CSV用的是完整捕获的launch归因。

## 6. 当前版本与未做的事

- 新39基线：源码`0689f7d…`，observer使用无副作用参数元数据，Dense/Final两条
  都通过无采集latent精确对照。原始v1及其勘误保留，不用于最终计费。
- 当前streaming示例来自此前已验证的153-frame系统路径，和batch基线不是同一
  执行配置；其完整视频收益另有独立重复实验，本次不以不同trace直接排名。
- Dense477是旧诊断；其旧预览曾漏像素归一化，已另有修正预览。本说明只用该
  trace看系统活动，不做视频质量判断。
- Dense477的旧detail主要包含与GPU工作关联的API；完整等待API诊断以新
  Dense/Final39和current streaming窗口为准。
- Nsys `.nsys-rep`、SQLite、完整call记录、shape计数与来源锁都留在工作区。
- 本次不启动新质量、Tether或算法矩阵；先等待用户审阅。

本说明依据[Perfetto官方UI文档](https://perfetto.dev/docs/visualization/perfetto-ui)
及[Trace analysis文档](https://perfetto.dev/docs/quickstart/trace-analysis)，并用
固定SHA的官方Trace Processor v58.2进行导入/SQL验证。没有把Chrome JSON
文件后缀当成“已经验证能打开”。
