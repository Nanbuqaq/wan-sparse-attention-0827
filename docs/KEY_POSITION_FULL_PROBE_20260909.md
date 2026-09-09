# 原生509：源内容 × 历史K位置

两个短视频门禁通过后，仅在原bead/seed20260913做以下同输入2×2。
既有related-original为`context509_local_v1/lane1/shot`，不重跑；新增三条：

- related-recent：40–47 source，虚拟时间88–95/current phase24。
- away-original：56–63可见away源，保留原位置。
- away-recent：同away源，虚拟时间88–95/current phase24。

均只替换native shot8，不移除初始global、不额外清上下文、不复制source；Attention上限32。
H2D源8一次，V和K的空间通道不变；仅recent做显式时间旋转，delta分别64/40，计时单列。
真实历史frame IDs保留，effective位置不同，route/storage version不同；不是layout优化。
源将在正常native返回pin/rolling下退出，因此首先看首返回状态，再看晚段，不假设永久驻留。

away源只是可见内容不相关，并非信息论上无任何目标信息：其深层KV可能已上下文化。
pre96 actual latent/noise必须匹配旧控制。数字/短gate不能代替完整视频审查。
本轮只有单seed机制探测，不进入正式holdout/Pareto，不扩量到未测配置。

代码只本地冻结；新的外部push仍等用户批准目标仓库，不绕过权限边界。
