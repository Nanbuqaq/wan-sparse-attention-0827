# Q摘要在何时拥有当前文本信息？

来源：已锁LongLive2 `6b36d20e...`，`wan_5b/modules/causal_model.py` L876/L906、L1532–1583，
`pipeline/causal_diffusion_inference.py` L554–574；只适用于本次无图像条件的T2V路径。

- 每个chunk首个denoise从该chunk的noise slice开始；patch embedding和time embedding先独立计算。
- 第0层先self-Attention，之后才用当前context做cross-Attention，随后FFN。
- 因而固定noise、timestep、geometry与RoPE/shot位置时，首denoise第0层self-Q不依赖当前文本内容。
- CPU测试执行官方forward AST并以明确test doubles替换SA/CA：改context不改变SA输入，但改变block输出。
  这是代码依赖/执行顺序验证，不是新训练权重GPU对照，也不是所有层Q的语义结论。
- 后续层已经历文本cross-Attention，后续denoise的输入latent也包含生成反馈，因此不能推广成“Q都无信息”。

研究影响：只给首步L0 Q摘要和同一候选原型的selector，在上述固定条件下不能区分两个不同当前文本请求。
它仍能用候选V/age元数据作内容无关选择；但不能因此称已实现当前目标语义路由。
候选下一步是当前condition摘要（仅当前，不允许访问预编码的未来prompt字典）、延迟到有语义的层、
或不同denoise阶段使用不同信息源。需要明确扩展因果输入契约并做隔离校准，不能用Dense输出补入口。
尚未实现/冻结新自动selector；本条不改现有negative cost model，不宣布新utility成立。
