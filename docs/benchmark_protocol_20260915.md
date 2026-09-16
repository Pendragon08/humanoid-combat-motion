# 基准驱动的实验执行补充（2026-09-15）

本文件补充 `step7_protocol.md` 的E1--E5，不改动固定控制与数据转换的既有定位。对应配置 `configs/benchmark_protocol_v1.json`；BM1已完成，BM2–BM4仍待输入与基线门槛；不能将旧开发结果改名为基准成绩。

## 基准与验收

|任务|数据|对照与输出|准备状态|
|---|---|---|---|
|BM1 通用生成|InterHuman官方split|InterGen / 当前TIMotion / ours；官方评价器、FID、文本匹配、多样性|已冻结并完成当前TIMotion与适配器各20轮；1099条可用子集，InterGen对照未执行|
|BM2 拳击反应|DuoBox|Ready-to-React、相同信息条件的M1；反应/双人持续/长时；在线离线分表|官方申请入口可得，本项目授权及本地数据未确认|
|BM3 事件与参考质量|DuoBox优先，Inter-X补充的独立人工标注子集|同候选raw/contact/temporal/both；事件、骨长、脚滑、速度、攻击者、覆盖|接触测试标注尚未建立|
|BM4 数据效用|独立机器人参考与配对事件|固定方法重训种子0/1/2；V vs V+generic vs V+ours，M就绪后剂量实验|既有门槛仍未通过|

Inter-X补角色与跨库，KungFuAthlete仅单人辅助；GTA Combat/Harmony4D第二阶段。DuoBox原论文63.4分钟、3人、120fps→30fps、80/20划分：保持官方协议复现，不称原协议为跨身份独立测试。测试可使用既有MoCap，训练和外部先验用量另行披露。

## 实施顺序

1. 旧E2两分片已确认完成，80个输出SHA256核验通过，结果见 `records/data_constraint_v2/summary.json`；不重复启动，也不把训练集弱几何诊断当成BM3基准成绩。
2. BM1冻结代码、split、权重、归一化、评价器和种子；官方全测试与自建格斗子集分表。现有开发数据不当最终测试。
3. DuoBox获许可后以官方脚本预处理；记录官方split和转换，保持两人共同坐标。先用官方权重验证接口，不先重训大模型。未经检查不得声称直接兼容TIMotion的262维表示。
4. 独立接触标注先做30--50片段开发审计；确认性测试数量与来源数在审计后确定，不用该数声称统计充分。目标事件分Hit/Block/Avoid，拳击优先，踢腿及反击以标签可用为前提。
5. 完成BM3和质量/覆盖选择；同预算、同保留量两种视角分别报告，保留失败/拒绝/未知。
6. 原有基线门槛、预算及独立测试冻结通过后做BM4。M未就绪只做V增强；无成功校准标签先不训练物理critic。

## 事件标注与信息约束

模板 `configs/benchmark_contact_annotations.csv` 只有表头，无伪造样本。事件字段包含来源、演员对、片段、攻防角色、目标、身体部位、窗口、方向、相对速度、置信度、证据、双人复核与版本。未知字段为空，不使用0代替未知。两名标注者独立判断；分歧复核并报告一致性。生成方法和分数对标注者隐藏。

任务条件与评价真值分开：Task Condition用于生成；Evaluation Annotation仅用于评测。不能给生成器测试反应真值。离线完整攻击序列与在线历史观察分表；两者都须匹配额外事件条件。在线归一化、特征、速度和预处理不得读取未来帧。

Hit检查部位/窗口/方向；Block检查先拦截且目标未中；Avoid检查窗口内间隙。骨架距离伪标签与人工确认、仿真接触、真实力测量分开。早终止后事件记censored/unknown，报告观测覆盖；完整配对任务成功仍把早失败计入分母。不要仅在幸存帧上计算总体表现。

## 清单与冻结

`configs/benchmark_manifest.example.jsonl` 为无记录空清单，需从真实数据填充。每条记录必须有sample_id、dataset、split、source_ids、source_sha256、motion_path、representation、fps、license_status、lineage_ids；test还需test_use=confirmation、untouched=true、independent_annotation=true和annotation_id。开发数据不能标untouched。

本课题来源划分按源录制/演员对建立，双方、相邻切片、近重复及所有生成后代同组。官方复现清单独立保存；官方划分不满足本课题来源隔离时报告差异，不自行改后仍称官方协议。评价输入允许测试攻击条件；合成训练样本只允许train父来源。

冻结包包括真实清单与独立标注文件的SHA256、官方提交、模型和评价器哈希、预处理、随机种子、容差和版本、数据/预训练重叠审计。准备检查器只读清单和输入文件，默认不启动训练。空清单/缺文件/错误哈希/跨split血缘拒绝确认性就绪。

## 对应模块改进

M1先通过DuoBox同信息反应评测；后续可消融generated-history训练。M2保持现有骨长与时序实现，在独立事件上检验方向/拦截顺序，避免修正导致速度、脚滑退化。M3先比较随机、质量、质量+覆盖；可靠执行正负标签充分再启用学习代理。LIMMT、OmniRetarget、ARMFlow、RobotMDM和CLAIMS只按原任务借鉴，不声称组合已有效。

## 官方来源

- InterHuman https://github.com/tr3e/InterGen
- DuoBox https://github.com/zju3dv/ready_to_react
- DuoBox论文 https://arxiv.org/html/2502.20370v1
- Inter-X https://github.com/liangxuy/Inter-X
- KungFuAthlete https://github.com/NPCLEI/KungFuAthleteBot
- GTA Combat https://liangxuy.github.io/actformer/
- Harmony4D https://jyuntins.github.io/harmony4d/

这些工作已在Overleaf参考文献中，不新增重复条目。所有新运行应生成独立run_id和配置/数据哈希，不覆盖历史输出。
