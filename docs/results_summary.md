# Results summary / 结果状态

## 本轮数据实验进展

E1种子来源与完整性审计在本地及nlp03完成：53条文件校验通过，53条结构/有限数值通过；训练45条/18个标题组，验证4条/1组，历史开发测试4条/1组。全部是单人视频恢复，尚无人工确认的配对接触真值或成本记录。标题组不是已核实独立拍摄场次。

结果和复核清单见 `records/data_sources_v2/local_20260915_v2/`；服务器完成回执见 `records/data_sources_v2/remote_observation.json`（原始服务器文件待下载）。原始姿态需要归一化，但旧导出已有这一处理，不把它作为既有失败根因。

E2计划在同一20对生成候选上比较原始/仅接触/仅时序/二者结合，80个输出。两GPU启动命令发出后ZeroTier会话断开，没有收到PID或完成回执，**状态未确认**。本轮没有新的下游策略训练成绩，也未证明数据效用；重连后必须先查任务，不能直接重复提交。

## 当前证据边界与修订

研究对象是示范数据，详细计划见 `docs/step7_protocol.md` 顶部。以下历史数字没有改变：

|证据|结果|允许的解释|
|---|---|---|
|第七步，三组各3种子重新训练|视频种子3.39%，普通增强0.78%，接触适配增强0.78%|低基线、单开发来源上的有限负面效用试验|
|TWIST冻结控制器|官方接口核验完成，配对生成严格成功低|控制器兼容性，不是数据训练价值|
|双人几何修正48对|穿插帧11952→1033；R0/R1双人严格成功均0/48|几何后处理改善部分指标，但未证明动态可执行；没有用修正数据重训|
|MoCap剂量与独立配对效用|未执行|不声称节省专业MoCap或获得完整双人学习收益|

来源：`records/step7/summary.json`、`records/twist/summary.json`、`records/paired_refinement_v2/summary.json`。新计划与历史实验编号分离；下文是既有结果，不是新协议成绩。

## 2026-09-15 历史第七步结果

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
