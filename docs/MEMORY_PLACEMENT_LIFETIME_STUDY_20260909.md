# 返回镜头的记忆使用：放置、寿命与容量

用户已恢复探索；导师系统审阅包和已关闭实验不改写。本轮是开发研究，
不是正式holdout，也不是声称自动在线准入成立。

## 冻结问题与实验

原生5B BF16/FA2、704×1280、128 latent/509 pixel，沿用已验证目标缺席的
toy/bead两开发任务、seeds 20260913/20260914、starts 0/24/48/96。
源为已提交的40–47 latent；返回文字不完全重述生成外观或实际珠子状态。
统一adaLN 16warps/1stage，新批同组共享noise及数值配置。

每组五臂，共20条，不重跑旧raw/log存储因子：

| 臂 | 注意力容量（latent） | 作用 |
|---|---:|---|
| none | 32 | 同数值配置对照，无CPU episode archive |
| global | 32 | 源8帧替换全局8帧，持续至结束 |
| global_one_chunk | 32 | 同global，但104开始前恢复原全局8帧 |
| shot | 32 | 同源替换上一镜头pinned8帧，原全局不动 |
| window128 | 128 | 原生接口扩大窗口，无外部episode hook |

global与one_chunk必须在104前全latent相同；恢复仅替换旧global，
不撤销已生成返回chunk或其clean KV。额外备份2.595GB及恢复传输/时间全部
计费，CPU峰值相应增加，不能宣称同总存储成本。shot是位置、被替换上下文和
自然寿命共同改变，不称纯layout。所有召回保留原absolute RoPE，包括原shot offset。

window128在128帧工作负载可保留全部KV，但改变早期轨迹及训练窗口设置。
只要求相同noise/config/骨干，按自身source评估，不要求prefix与32窗口相同。
原生代码仍分配positive/negative两套KV，约83.05GB；另加模型与临时量，
预计H200141GB可承载，H80080GB可能OOM。以实际型号/峰值/终态为准，
不使用fallback，不以硬件不足记算法质量失败，不把容量估计当实测。

## 门禁与判据

1. CPU单元测试；本地4090真实GPU走通新增one_chunk和128窗口分支。
   64 latent/256×512只作技术门禁，不做质量推广。
2. 验证global与one_chunk的源/首返回精确一致及恢复流量；确认窗口128确实分配。
3. 测试、dry-run、push后，一个4-lane冻结批次；每lane一个prompt/seed，
   五个不同arm依次运行，交错正反顺序，失败隔离，无自动重试。
4. 视审所有自身source、离开末段、首次返回、晚段及完整季度板。
   单独记信息回忆、场景污染、动作/状态正确性、画质；不能用部分信息回来代替
   整体胜出。若window128的目标未真实缺席，标workload-invalid。
5. 若one_chunk降低晚段污染且保留信息，才进一步研究因果记忆退出；若shot有效，
   再做位置/上下文更细消融。全部不成立则保留negative，不自动加权调参到好看。

本轮先隔离机制，不引入mask、teacher在线查询、重定位RoPE、LoRA训练或复杂utility。
