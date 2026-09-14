# Experiment log / 实验记录

## 2026-09-14 paired-v1

1. 已审计本地和服务器现有代码、方案文档、GPU、空间和 Python 环境。
2. 核验 TIMotion 与 BeyondMimic 官方源代码并固定提交；发现当前公开 TIMotion 不是 RWKV 版本。
3. 在现有 MuJoCo/Torch 环境（spec hash 457e8491，未修改）执行 `bash scripts/run.sh -m unittest discover -s tests -v`。
4. 14 项测试通过：原有 3 项机器人约束测试 + 新增 11 项接触/协议测试。耗时约 1.16 秒。它们属于单元测试，不是正式对比实验。
5. 官方模型和数据仍在获取中；首次 Python 下载没有有效数据进展，停止后改用限时、限速检查的 curl 下载。下载文件尚未通过完整性验证前不得使用。

未执行项目：正式 TIMotion 指标复现、格斗子集微调、双机器人学习型跟踪、物理 critic 训练、三种子下游数据效用比较。
