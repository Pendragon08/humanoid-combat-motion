# 原始协议字段解释（不覆盖冻结协议）

`preparation_foot_penetration_before_m`：整体根高度对齐后的中间参考，在准备阶段帧集上的最大胶囊穿地代理量；不是未修改原始参考。

`preparation_foot_penetration_after_m`：当前组候选在同一准备帧集上的量。

由于高度对齐为常量平移，原始准备阶段最大穿地代理量为 `max(0, preparation_foot_penetration_before_m + initial_ground_shift_m)`。本轮四条参考的原始量均非零。使用该原始基线比较时，四个 `prep_support` 候选的最大穿地量都变差；没有相对于原始数据的几何改进证据。

原始 `support_analysis.json` 成功末帧哨兵解释错误已由 `support_analysis_v2.json` 明确纠正。评估器 `metrics.json` 的 `first_*_failure_steps` 仍保留历史格式：末帧可能为未失败哨兵，需结合逐帧谓词/成功状态解析，不能独立解释为失败事件。

这些说明不改变成功阈值、不覆盖任何执行结果、不把变化后的策略状态作为参考。
