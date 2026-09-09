# 相同逻辑区域的物理组织实测

不是新视频质量实验。固定已有source红色区域685/7040 tokens，使用已验证reset-reveal
capture的L0/phase0实际CPU K/V，24heads、head128、BF16，源8帧×880token。

- 6种物理组织：exact、within-frame Block64、Page256、spatial4×4、spatial8×8、Frame880。
- 3种消费路径：需求时打包成一次fused copy；K/V分别pack/pin/copy；预先pin整个archive后按连续run复制。
- 每次GPU整理后必须与同一原始ROI K/V逐位一致；logical SHA不变，padding/额外取回不进入逻辑输出。
- 布局构建与完整archive pin单独计费；raw identity布局直接复用已有source，不做无谓复制。
- demand完整计入CPU pack/pin、copy提交、等待及GPU gather。CUDA记录的是stream跨度，可含CPU提交间隙，不称纯copy engine忙时。
- 5轮全网格warmup、30轮固定随机顺序；逐点median/p95及每次记录保留。没有自定义JIT，compile记null。
- 所有布局的pinned archive同时驻留；另按两套不同staging形状保守计入请求字节，默认2GiB张量形状预算。不是实测allocator reserved/RSS。
- 既有capture读取/取出CPU source作为setup；不把它误算为测得archive生产D2H。完整创建—消费收益还需按重访次数摊销。
- 纯单层replay，不把结果乘30冒充端到端视频，不以异步copy API声明overlap，不改变Attention dataflow。

代码CPU布局测试通过后先在GPU0做本批正确性及测量；GPU1继续较静止源最后一条。
所有方式给相同源、相同逻辑ROI和通用pack机会；允许exact staging胜出，不预设空间布局赢。
