# 初始状态是否应继续充当永久锚点？

动机：本地8-case没有整体赢家；18条实际Attention Q/K/V/O的FP32回放通过全部门禁。
bead reset在末次去噪layer14的采样质量约initial25.1%/source11.8%/current63.1%，
layer29约current96.9%。这是固定几何query样本，不是全Q均值，也不是语义因果证明。

当前进一步做四条原生509开发视频，只针对bead/seed20260913：

- source_only：初始空罐global退出，只把已提交source8置于global，初始有效历史8。
- source_repeat：初始global退出，同一source8置于global与shot，共16逻辑历史帧。
- 每种都有相关40–47与错误56–63源，两者都是过去且返回时非resident。

全都保留原absolute RoPE和当前global时钟，不重新旋转；max物理窗口32、positive-only
公共分配、BF16/FA2/Triton3.2、adaLN16/1相同。source_repeat是显式重复的逻辑边：
相当于对source加log2先验，不是padding。实际H2D仍只有source8（2.595GB），
另外2.595GB的D2D复制单列；source_only没有此复制。不能称两种相同RoutePlan。

首返回source_only实际K16，下一chunk K24；source_repeat为K24→32。
短GPU门禁只做正确性，不用于质量。完整视频复用已完成的none/shot/reset_keep_global
作为同GPU族、同输入、同compiler对照，但必须实际验证pre96；不重复原8条控制。

只有预定单次返回后的32 latent被覆盖，未验证下一次场景切换时如何再次退出，
也没有自动source选择。即便正结果也不能直接声称一般在线方法成立。
