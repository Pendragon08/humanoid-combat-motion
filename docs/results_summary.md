# Results summary / 结果状态

## 2026-09-15 当前结果

第七步首轮组件实验已完成：3个数据组×3个控制器种子、每次12,288,000环境步，共36条留出评估记录。严格跟踪通过率（均值±种子间样本标准差）：视频来源3.39%±0.90%，加普通生成0.78%±0.78%，加适配器与几何接触优化0.78%±0.00%。当前未发现合成数据收益；4条开发留出动作仅来自1个视频标题组，不能作为最终独立确认。几何接触优化组未通过可靠物理筛选，且本比较只验证单机器人跟踪。

完整物理critic（物理评价网络）仍缺可靠标签。固定控制器覆盖预测及K=6历史排序已执行，但预测误差接近常量基线、候选均提前失配，不允许用作物理可执行性认证。38个单元测试及最终产物校验通过，只说明对应软件与证据检查通过。最新记录见 `records/step7/summary.json`、`records/step7/artifact_check.json`，完整历史见 `reports/experiment_progress_20260914.html`。

## 历史启动记录（下表不是当前进度）

| Experiment | Status | Meaning |
|---|---|---|
| Earlier single-person pilot | completed, negative physical result | 不能说明 synthetic data 提升 imitation |
| Contact/provenance unit tests | 18/18 passed including legacy tests | 仅验证软件规则；球体代理不等于真实接触力 |
| TIMotion official inference | pending weights/environment | 尚无生成结果 |
| Official FID/R-precision | not run | 不能以自定义特征或论文数字充当复现数值 |
| Combat adapter + contact guidance | in progress | 未完成训练 |
| Paired robot tracking | not run | 需要 retargeting + 学习型 tracker |
| Physics critic | not trained | 缺少真实 rollout 标签 |
| Real vs Real+TIMotion vs Real+Ours | not run | 核心有效性结论尚未确定 |

本轮不做“方案已验证可行”或“提高成功率”的结论。达到该结论需要完成用户计划的模型、物理和下游学习实验。

Isaac Sim binary physics: **passed** (240 real simulation steps; cube landing). Isaac Lab G1 task dependency installation remains in progress. This result is not humanoid motion tracking or synthetic-data utility.
