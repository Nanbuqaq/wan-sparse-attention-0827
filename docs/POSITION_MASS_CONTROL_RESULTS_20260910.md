# 相同总Attention概率，仍可能读出不同的记忆

CPU-only分析已完成，9条原始FP32/native回放门禁全部通过。固定原始Q、V和其他区域K，
只给source时间K做已注册delta64重定位。对每个采样head/query，用
`bias = logZ(source,new) - logZ(source,old)`匹配source的总Attention概率。
该bias是完整teacher信息拟合的离线对照，不是在线selector。

角色总概率最大误差为0，但原source内部的权重分布没有因此变成重定位后的分布。
六个denoising记录中，source内部概率TV均值0.236–0.506；匹配总概率后的输出残差，
相对总重定位输出变化的范数比为0.251–0.659。末次denoiseL14的比为0.659。

精确分解是：

`O_new - O_old = Σ_r (mass_new[r]-mass_old[r]) · O_old[r]`
`                 + mass_new[source] · (O_new[source]-O_old[source])`

第一项改变“分给各组多少”，第二项改变“从source内部读出什么”。两个向量可能抵消，
范数比不是可相加的解释比例，更不是质量改善百分比。

只在first-denoiseL0，输入条件与此前实际重定位轨迹完全一致；其source概率也复核匹配。
后续记录是固定旧Q反事实，不是新生成轨迹。每头32个几何query、全部24heads，非全Q/语义区域。
clean commit三行单列，不是第五次去噪。木箱负结果仍保留：内容影响不同，不代表状态关系正确。

事实：`results/metrics/memory_activation_20260910/position_mass_control_v1/`，含全部行、固定gain2/4
对照、误差、SHA和解释。没有新GPU生成，也没有在线策略据此冻结或推广。
