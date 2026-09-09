# 延迟分叉：源副本应该被滚走，还是继续作为锚点？

初始锚点四条视频技术通过。相关source_repeat恢复了红珠，source_only及错误source
未恢复；但相关repeat晚段有珠子移动/散落，状态仍不稳定，不作整体推广。

源码滚动规则：source_repeat第一返回chunk完成后，native把新生成的返回chunk设为
shot pin；当第三个返回chunk到达时，第二份原source被正常rolling驱逐，source先验
由两份降到一份。这提供一个独立的晚段机制假设，不以注意力质量直接证明因果。

新增两条本地原生509：bead/seed20260913，相关及错误源，policy=source_repeat_pinned。
只在首返回clean完成后把pin从新生成的返回块改回第二份source；不复制新的KV。
首/次返回Attention内容应不变，所以必须完整pre112 latent与对应natural-repeat控制
逐位一致；前445个解码pixel frame也应相同。分叉只能出现在第三返回chunk之后。

H2D仍2.595GB、原source重复D2D仍2.595GB；额外仅30层的两个int64 pin元数据写入，
计费单列。所有Q/K形状与natural-repeat保持一致，不能归因于更大Attention预算。
此操作同时选择长期source与被放弃的首返回锚点，不能拆成纯layout收益。

单次返回后的四chunk机制试验；未验证后续新场景/状态写入，不能称一般在线方法。
两条完整运行兼作新pin分支的真实GPU门禁，失败保留，不进入正式扩量。
