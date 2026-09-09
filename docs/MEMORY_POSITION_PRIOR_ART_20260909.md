# 位置绑定的已有工作与本轮可研究问题

公开源码访问与SHA核验：2026-09-09 UTC。以下是代码事实，不是本项目复现结果。

- **StreamingLLM**，`mit-han-lab/streaming-llm@2e5042606d69933d88fbf909bd77907456b9b4dd`。
  [modify_llama.py L94–103](https://github.com/mit-han-lab/streaming-llm/blob/2e5042606d69933d88fbf909bd77907456b9b4dd/streaming_llm/pos_shift/modify_llama.py#L94)：先保存未施RoPE的past K，再按cache内位置对消费K施旋转。
  所以“历史位置不必等于原始绝对时间”不是新概念；本轮在已舍入BF16 K上做相对旋转，和raw-K重施RoPE不逐位等价。
- **InfLLM**，`thunlp/InfLLM@12b70798f56e56ebb23c53c7018091a3f540a028`。
  [context_manager.py L744–747](https://github.com/thunlp/InfLLM/blob/12b70798f56e56ebb23c53c7018091a3f540a028/inf_llm/attention/context_manager.py#L744)：global Q使用`n_local`的同一旋转角，global记忆不逐项保留真实年龄距离。
  同文件L480–530先算local部分，再取global块并追加Attention，包含可选global stream、cache和同步。
  这是代码的分阶段供给/消费设计；未在本项目测该实现，不能把注释“overlap”当实测收益。

**对本研究的影响**：不把recent重绑定本身作为新算法贡献；不把任意位置策略的单seed红珠恢复作为“更快更好”。
视频独有的待验证张力是：镜头切换应隔离旧场景，但已提交的身份/状态仍需要被读取；同一位置处理可能同时激活正确状态和错误场景。
下一步的可区分机制是“选哪组信息、怎样赋予消费位置、何时释放”，而非统一把所有旧KV拉近。
系统上，若角色/区域已因果分组，可比较分组布局、有限驻留与local/global部分Attention流水；必须计入组构建和完整关键路径。
当前四格只是指定source下的因果干预，新seed复验、距离/镜头偏移分解、静止源与跨主体验证均不能省略。
