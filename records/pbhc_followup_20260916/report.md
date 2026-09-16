# 完整拳脚动作训练与后续诊断报告

4条动作各完成2,000次更新、49,152,000环境步，总196,608,000步。全部16组阶段评测与12组扰动诊断已收集。
这是单参考开发校准，不是独立泛化测试，也不是合成数据效用证明。原始随机化结果保留为主结果；无扰动结果只解释失败来源。

|动作|评测条件|更新次数|完整执行率|逐帧严格通过率|根误差(m)|
|---|---|---:|---:|---:|---:|
|Hooks_punch|no_push|2000|90.62%|6.25%|0.2246|
|Hooks_punch|nominal|1001|100.00%|0.00%|0.0773|
|Hooks_punch|nominal|2000|100.00%|0.00%|0.1355|
|Hooks_punch|randomized|0|0.00%|0.00%|0.1299|
|Hooks_punch|randomized|501|65.62%|0.00%|0.2323|
|Hooks_punch|randomized|1001|81.25%|0.00%|0.2846|
|Hooks_punch|randomized|2000|84.38%|0.00%|0.2586|
|Horse-stance_punch|no_push|2000|96.88%|6.25%|0.2535|
|Horse-stance_punch|nominal|1001|100.00%|0.00%|0.2961|
|Horse-stance_punch|nominal|2000|100.00%|100.00%|0.0619|
|Horse-stance_punch|randomized|0|0.00%|0.00%|0.2053|
|Horse-stance_punch|randomized|501|18.75%|0.00%|0.2658|
|Horse-stance_punch|randomized|1001|68.75%|0.00%|0.3289|
|Horse-stance_punch|randomized|2000|84.38%|0.00%|0.2656|
|Roundhouse_kick|no_push|2000|9.38%|0.00%|0.3157|
|Roundhouse_kick|nominal|1001|0.00%|0.00%|0.2009|
|Roundhouse_kick|nominal|2000|0.00%|0.00%|0.1215|
|Roundhouse_kick|randomized|0|0.00%|0.00%|0.1592|
|Roundhouse_kick|randomized|501|0.00%|0.00%|0.2207|
|Roundhouse_kick|randomized|1001|0.00%|0.00%|0.2670|
|Roundhouse_kick|randomized|2000|0.00%|0.00%|0.2374|
|Side_kick|no_push|2000|84.38%|0.00%|0.4774|
|Side_kick|nominal|1001|28.12%|0.00%|1.0819|
|Side_kick|nominal|2000|100.00%|0.00%|0.4172|
|Side_kick|randomized|0|0.00%|0.00%|0.1621|
|Side_kick|randomized|501|0.00%|0.00%|0.2157|
|Side_kick|randomized|1001|21.88%|0.00%|0.4935|
|Side_kick|randomized|2000|90.62%|0.00%|0.4793|

## 解释边界

- 完整执行只表示未触发原环境终止，不能代替严格跟踪。
- 根误差均值仅统计首次终止前且包含终止帧；必须同时看完整执行率，避免将早失败造成的低误差当作更好。
- no_push只去掉外力；nominal进一步去掉材质/质心/默认关节及初始扰动，保留参考初始速度。不是重新训练控制器。
- nominal提高只能说明模型对扰动敏感，不能将其成绩冒充原随机化任务成绩。
- 各动作仅一个训练种子；32环境不是32个独立动作来源。
- 后续数据修正、生成器训练和数据效用对照需结合本报告判断条件，不因进程结束就宣布基线合格。

四条回放均固定环境0，展示实际记录状态，首次终止后冻结；不按成功结果选择样例。
