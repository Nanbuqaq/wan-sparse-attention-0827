# 位置读取的跨主体开发验证

红珠填充中两seed及较静止源两seed已支持条件性类别恢复，仍非自主方法。
下一步固定旧toy/seed20260913，测试是否只对红色填充量任务有效。

- 原生509、已有fb1afb0代码与真实GPU门禁、相同BF16/FA2/Triton3.2/adaLN recipe。
- 复用`context509_local_v1/lane0/shot`作为related-original，不重跑。
- 三新增格：related-recent、away-original、away-recent；同source40–47/56–63、同delta64/40。
- GPU0一条related-recent；GPU1两条away，全部事先冻结，不按第一条结果选择后续。
- actual pre96 latent/前381 decodedRGB、noise、来源、H2D与容量必须配对。
- 首/晚返回分别检查独特脸、涂色/材质、身体构造、背景污染与多余部件/动作。
- 这是同类调参外的对象类型验证，但仍是已看过Dense的开发prompt，不叫正式holdout。
- 不把只回来一种颜色/脸部细节当整体身份成功，不新增seed追正结果。
- 仍指定source与返回时机，不是自动selector。
