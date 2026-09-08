# 原生分辨率场景退出小实验

H20固定批次仍保留原SHA，因H节点心跳中断被平台自动退回队列，不重投。
两张本地4090执行一个不同硬件/分配策略的8-case开发批次：toy/bead × seed20260913
× none/shot/reset_reveal/reset_away。所有配置均获得CFG1 positive-only公共容量优化。
使用原生704×1280、128 latent/509 pixel、local32、固定adaLN16/1、Triton3.2。

动机来自已完成旧global召回的污染，以及首返回源码frame组成（与实测pin位置一致），
不是低分辨率视审。reset的相关与错误源短门禁均已通过实际pre48 latent对照；
FA2实际Q1024/K3072×150calls，下一chunk K4096×150calls，时钟不变。

四臂：无召回；相关源放入上一镜头槽位但不额外退出旧上下文；同源＋退出；
等字节错误旧源＋退出。通过native local_end排除无效suffix，不将零KV加入softmax。
源40–47或错误56–63均已提交，所有原absolute RoPE保留。回忆时机与源是实验指定，
不是自动在线selector，也不是正式holdout。

分配优化只证明了低分辨率逐位等价。完整local32生成峰值预计接近4090容量边界，
故每条lane先运行none，实际完整基线通过才顺序运行其他三臂；失败不fallback或改分辨率，
该lane其他臂记未启动/不可做质量判断，另一lane独立继续。窗口128不声称能放进4090，
其H200对照仍等H20；不能以这里32窗口胜出宣布超过所有容量强基线。

reset保留8global+8source作为历史，加当前8，首次Attention实际24帧，而普通路径32帧。
pre-return时失效16帧suffix，其中8帧本来会被正常rolling驱逐，净Attention减少8帧。
后续恢复正常32上限。记录实际shape，不把原分配不变写成逻辑选择相同。

技术终态之后审查各自source、away末段全部128帧、首返回、晚段和季度板。
分别看信息恢复、场景污染和正常画质；不能因回忆了红珠就忽略额外绿珠/花朵。
这是single-seed机制初筛；即便两个任务改善，也需独立seed/长视频和自动准入验证。
