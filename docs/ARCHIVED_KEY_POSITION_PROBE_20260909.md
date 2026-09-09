# 历史K的位置重新绑定：算法探测，不是搬运等价

可见状态的双seed cut消融中，单独去掉切换文字未阻止清空/减量；保持source相位则保留
更多原红珠。该结果限定于同场景可见控制，不能直接推广到真正离开后的retrieval。

隔离capture上的位置重放同时显示：只校正scene phase不一定增加source质量；把source
映射到current之前的最近虚拟帧有不同的局部输出作用，但这些量不是视频质量。

因此先做新GPU门禁：保留选中source的真实历史frame IDs、全部V和K的空间通道，
只对已储存的BF16 K施加相对时间旋转。Wan5B head128中仅前44个real通道为时间RoPE。
冻结linear scale1/theta10000，virtual帧为target-8…target-1，使用current scene phase。
这不是再次施加完整RoPE，也不恢复量化前raw K；新position policy和storage-version SHA明确记录。

先验证CPU/GPU相对旋转一致、delta0严格identity、空间通道不变、FA2输出对FP32通过。
再各运行一个相关/away source的64-latent真实分支门禁。原global/shot/current逻辑容量32不改，
V不变，H2D只一次，GPU变换的时间/输出字节另计；transform字节不等于实测全部HBM流量。
只有技术与pre-return检查通过后才考虑原生509。未经视频验证不冻结任何线上位置策略。

所有新代码与结果保持本地；GitHub推送仍等待用户确认，未绕过权限拒绝。
