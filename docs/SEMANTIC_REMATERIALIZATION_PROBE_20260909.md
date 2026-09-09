# 状态与旧动作条件：隔离latent重编码

源权重两份可激活红珠，但旧动作/散落仍在；保持第二份源pin也未消除晚段动作。
预112的严格配对通过，因此不能再把动作起因简单归于第三chunk时源副本被驱逐。

下一小试验不是复现原KV，而是把已完成source8 clean latent重新编码成当前可用KV：

- isolated-past：空历史下，使用原source条件，单个t=0 forward。
- isolated-current：同一latent、同样空历史、同位置，用当前return条件，单个t=0 forward。

两个版本对比隔离当前条件的作用；它们相对raw KV都同时改变了历史上下文构造，
不能将raw-vs-current差异仅归因于文本。旧1.3B pre/post recache实验的negative保留，
这里是5B有效缺席任务、同source latent上的新受控构造。

都使用source_repeat_pinned逻辑选择。原active context已被该政策判无效，所以直接
复用其GPU KV buffers作为重编码工作区，不分配第二套KV。原frame/source RoPE保留，
重编码后恢复当前absolute clock，并失效cross-attention缓存，使当前prompt重新生效。
预算16MiB CPU张量；source latent、原条件、当前条件hash的额外D2H及所有重建时间均记录。
版本SHA同时包含latent、使用的条件、source坐标、空历史构造与位置设置。新KV不是原KV。

先本地64-latent GPU门禁，检查pre48、buffer指针复用、source8占据、当前clock恢复、
一次重编码、实际Attention shape/pin以及显式字节。通过后才考虑两条原生509控制。
仅过去已完成latent和当前已知条件；没有新训练、VAE或外部模型。源选择仍是实验指定。
