# 当前条件驱动的简单因果场景基线（准备阶段）

目的不是把T5最近邻称作新算法，而是去掉当前位置机制实验中人工指定source frame和target frame的依赖。
只适用于当前有显式分镜/返回文字的开发协议，尚非通用实体跟踪、正式25%稀疏方法或论文胜出。

预注册实现：每次观察到原生shot phase切换，保存刚结束场景最后已提交的8帧原KV及其当时T5摘要；
不预先知道目标何时返回，不读未来帧或未来文本。CPU FIFO归档预算8GiB，原生active KV仍32帧。
在线selector只接收当前文本/当前T5摘要和过去descriptor，不接收完整K/V或Dense teacher。
当前指令有明确`Back to`/`Return to`等前缀时，在距离当前≥32帧的过去场景中选择。
固定规则：masked-token mean、cosine floor0.80、距最高相似度0.05以内取最新提交；参数来自已标注的开发表征，不叫独立holdout选择。
归档相似度可混淆多个同类对象，前缀识别也有限；这些限制必须保留。
选中后才取该bank原KV，复用已验证的recent时间绑定；绝不让selector偷看bank张量。

因果输入契约新增**仅当前condition摘要和当前原始文字**。即使驱动预编码了所有prompt，selector也不得取得未来字典。
清洁提交之后才记录过去描述符；本次当前文本不是已观测attention统计，可在当前调用前使用。
archive/摘要D2H、选择、H2D/rebind均计入完整generation计时，额外归档开销不得藏掉。

先做64-latent真实GPU gate，对既有privileged recent gate完整latent/RGB验证；source/frame判断只作离线验收。
若源选错或数值轨迹不同，保留失败并查因，不替换成人工source fallback。
通过后才考虑完整视频等价验证。运行模块现已实现，CPU已检查归档所有权、FIFO预算、自动源选择和同phase只安装一次；真实GPU gate尚待运行。
