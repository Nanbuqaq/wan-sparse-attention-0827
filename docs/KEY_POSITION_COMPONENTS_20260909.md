# 历史位置分解：两个时间差分量，不是两种独立RoPE编码

首轮recent将source40–47/phase8映射到virtual88–95/phase24，总delta64。
其中48来自帧时间距离，16来自native切镜头偏移。它们加在同一44个时间通道上，
并不是能分别识别语义/时间的独立特征。共同旋转Q/K的内积不变；只旋转历史K则改变边。

在新seed四格复验运行期间，仅实现下列固定分解的CPU测试和GPU数值/短视频门禁，
不修改正在运行的fb1afb0工作树或新seed的策略，也不根据新seed结果调delta：

| 正确源策略 | 有效frame起点 | 绑定phase | delta |
| --- | --- | --- | --- |
| original（已完成/复用） | 40 | 8 | 0 |
| phase_only | 40 | 24 | 16 |
| age_only | 88 | 8 | 48 |
| recent_virtual（已完成/复用） | 88 | 24 | 64 |

不使用Dense teacher挑旋转角。不是寻找自由旋转参数最优值，只问既有delta中哪个结构性
分量有作用。错误源分别delta8/32/40。真实frame IDs不改变，全部V/空间K不改变，
归档字节和当前prompt/Q不改变；区别写入admission与KV storage version。

若先前结果能复验，短GPU门禁通过后只补正确源的phase_only/age_only原生视频，
优先seed13，再决定是否需要seed21；不重复四格、不增密度、不做自动selector的胜出声明。
判读源类别/材质、场景侵入、首返回与晚段动作，不能用Attention质量越大越好来选择。
两个分量非线性合成，单工作负载结论不能推广为关闭所有Narrative RoPE。
