# 新状态类别：先Dense可行性，再独立记忆验证

现有红珠开发结果不能用于宣称跨状态类别。冻结蓝色画布新文本与seeds20260923/24，
当前只执行两seed×revisit/visible-control的4条原生Dense视频，不运行memory方法。

- 0/16/32/64/96五阶段：空白→蓝色条带→较长停止保持→离开/可见控制→返回。
- 64前两分支完全同prompt；检查actual pre64 latent和前253 decoded RGB精确匹配。
- source为56–63；需蓝色区域和剩余白色区域同时可见、无重置，并逐帧看停止情况。
- 真正离开分支检查pixel285:381目标缺席；visible故意不离开。
- 不要求Dense返回必须失败；不根据memory输出挑seed。source不合格保留负面筛选结论，不直接扩方法。
- 当前selector代码SHA、floor0.80/margin0.05/minGap32与recent绑定已在配置冻结；新结果不回流调参。
- 这是独立的新状态类别开发验证，不混入旧1.3B四prompt正式holdout均值。
- gate使用原生509/704×1280；不缩成低分辨率假可行性。

新状态文本的停止保持比红珠更长，避免把正在画画的动态状态误当静止真值。
若通过，再明确记录冻结Dense审查SHA后决定验证；当前代码拒绝给蓝画布附加memory。
