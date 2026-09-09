# 未离开也会丢状态：继续分离cut与文本指代

settled-state首批4条Dense执行通过；source40–47已大体稳定。
但带相同cut边界的visible-control中，seed19红珠丢失、seed20填充高度明显降低。
因此当前不能把全部差异归为长间隔历史检索，不能据此推广新memory方法。

追加四条Dense-only（同seeds20260919/20）并复用旧cut+anaphora控制：

- nocut-anaphora：48以后与visible-control完全相同文字，只去掉48/96的scene prefix。
- nocut-explicit：48以后继续使用32时已经生效的“red glass beads fully settled”原条件，不再改prompt。

两者都不离开目标。前48 latent和初始noise必须与旧控制一致。
第一组比较是native切镜头整套机制（prefix文字/RoPE/pin）差异，不是单独RoPE消融；
第二组对比文本指代/条件改变，也不能当成“无文字帮助”的视觉记忆成功。
所有配置只用于测试模型的状态持续能力；source数量仍不在文字中重述。

代码和结果仅在本地冻结。外部push已被权限审查拒绝，等待用户明确授权目标仓库；不绕过。
