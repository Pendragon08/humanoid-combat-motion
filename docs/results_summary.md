# Results summary / 结果状态

| Experiment | Status | Meaning |
|---|---|---|
| Earlier single-person pilot | completed, negative physical result | 不能说明 synthetic data 提升 imitation |
| Contact/provenance unit tests | 14/14 passed including legacy tests | 仅验证软件规则；球体代理不等于真实接触力 |
| TIMotion official inference | pending weights/environment | 尚无生成结果 |
| Official FID/R-precision | not run | 不能以自定义特征或论文数字充当复现数值 |
| Combat adapter + contact guidance | in progress | 未完成训练 |
| Paired robot tracking | not run | 需要 retargeting + 学习型 tracker |
| Physics critic | not trained | 缺少真实 rollout 标签 |
| Real vs Real+TIMotion vs Real+Ours | not run | 核心有效性结论尚未确定 |

本轮不做“方案已验证可行”或“提高成功率”的结论。达到该结论需要完成用户计划的模型、物理和下游学习实验。
