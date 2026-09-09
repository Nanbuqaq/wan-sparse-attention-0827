# Dense可见状态：切镜头语句与RoPE跳变的2×2

前序同seed控制显示：切镜头可见控制会清空/降低红珠状态；不切镜头则保留红珠，
但物理运动仍会出错。因此此阶段只分解切镜头机制，不称长期KV准入方法。

source、noise、0–48 latent和模型固定；复用已有cut+anaphora两条作为完整native控制。
新增三臂×seeds20260919/20，共六条原生509 Dense：

- strip_words：仅在48/96将送入T5条件的`The scene transitions. `替换为已独立编码的
  同句无prefix版本；raw prompt列表、native cut检测、pin和RoPE跳变都保持。
- freeze_rope：保留raw/text prefix和native pin，只在48以后保持最后已观测的source
  RoPE相位8，不再使用16/24；旧KV不重新旋转。
- strip_words_freeze_rope：同时应用两项。

不是修改官方源码；通过明确adaptor记录condition aliases与每chunk实际相位。
仅限`settled_bead_visible_control`，不允许混入memory干预。每条必须实际pre48
与旧控制相同；原生pin事件仍为24/56/104。此处不能把prefix整体效应与单独RoPE混淆。

两seed、同card按相反顺序执行；一次本地冻结批次。CPU测试与dry-run完成后在本地执行。
GitHub推送和新InferHub提交仍因权限审查暂停，等待用户确认；不绕过该限制。
