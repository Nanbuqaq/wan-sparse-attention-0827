# 只改停止措辞的Dense控制

首轮4条形成蓝白图案，但长hold后工具仍活动，未达到干净稳定source前提。
不把这个失败当成新memory方法的失败/成功，也不要求新Dense返回失败才能入选。

新4条复用seeds23/24 × revisit/visible-control，只在latent32将第一句替换为：
`Only the canvas and its wooden easel are visible in the quiet studio.`
原句为`The painter and roller have left the view.`；其余hold、away/return文字完全不变。
不重述颜色，不新增cut，仍为0/16/32/64/96；selector SHA和参数继续冻结。
需actual pre32 latent/前125 decoded RGB匹配相应旧分支；source/停止和后续图案分别审查。
若措辞就能改善停止，应如实归因；若仍失败，保留源状态与动作纠缠的机制问题，不强行当holdout。
所有新运行仍Dense-only。没有SAM2、memory干预或训练。
