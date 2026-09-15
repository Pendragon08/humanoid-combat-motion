# 四项动作跟踪诊断（2026-09-15）

四项训练和32组最终评价全部完成；每项种子0、512环境、1000更新、12,288,000实际环境步，总49,152,000步。

- `summary.json`：来源哈希、训练预算、实际片段采样步数、逐参考结果与宏平均；仅完整八组评价才标记对应作业完成。
- `snapshot/INDEX.json`：116份小型配置、完成凭据、评价与压缩日志的来源及哈希。日志为归档时快照。
- `snapshot/tracking_diagnostics_20260915_refs/protocol.json`：冻结的参考选择和环境规格。
- `snapshot/tracking_diagnostic_video_*/provenance.json`：视频实际轨迹来源、固定环境0与冻结规则。
- `launch.json`：历史队列进程号，不表示进程仍在运行。
- `FINAL_REPORT_READY`：服务器结果汇总完成时间。

视频和中文解释位于 `reports/experiment_progress_20260914.html#tracking-diagnostics-results`。原动作文件和模型权重保留服务器的 `runs/tracking_diagnostic*` 目录，本记录归档不包含这些大文件。

四参考均匀／自适应的训练参考严格通过率为26.56%／25.00%，开发留出为0.78%／3.13%。这不是统计显著性结论：每条件只有一个训练种子，开发四片段同属一个来源组。单参考两组均为0%，其中四条选定参考只有第一条在训练中出现。尚不能据此扩大合成数据效用实验；本轮也未训练生成器或验证双人接触。
