# Humanoid Combat Motion Generation

<div align="center">

**面向人形机器人模仿学习的格斗动作数据生成实验**

[结果概览](docs/results_summary.md) · [实验报告](reports/experiment_progress_20260914.html) · [PBHC 基线](reports/PBHC_OFFICIAL_TRAINING_RESULTS_20260918.html) · [实验协议](docs/step7_protocol.md)

</div>

本仓库探索利用公开视频恢复动作和交互动作数据，构建单人及双人格斗参考动作，并通过重定向、物理诊断和模仿学习评价其可用性。包含轻量扩散生成、双人条件适配、接触几何修正、G1 跟踪及 EngineAI 平台探索等阶段性实现。

```text
视频恢复动作 / 双人交互数据 → 条件生成与动作增广
                         → 接触与时序修正
                         → 机器人重定向
                         → 仿真执行 / 固定跟踪器
                         → 数据训练价值评价
```

## Project Status

- [x] 完成单人生成先导实验、部分双人生成与几何修正实验。
- [x] 保存三个数据配置、三个控制器种子的历史训练价值对比。
- [x] 完成 PBHC / KungfuBot 单动作控制基线训练与评价。
- [ ] 尚未证明合成数据提高独立测试下的模仿学习表现。
- [ ] 尚未完成可靠双人格斗接触、跨来源泛化或真实机器人验证。

这里记录的是阶段性研究与复现结果。单人控制成功、双人动画和几何接触改善分别对应不同证据，不能合并称为“双人机器人格斗系统已完成”。

## Table of Contents

1. [Setup](#setup)
2. [Phase 1: Data Preparation](#phase-1-data-preparation)
3. [Phase 2: Motion Generation](#phase-2-motion-generation)
4. [Phase 3: Tracking and Evaluation](#phase-3-tracking-and-evaluation)
5. [Results](#results)
6. [Repository Structure](#repository-structure)
7. [Documentation](#documentation)
8. [Acknowledgements and License](#acknowledgements-and-license)

## Setup

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Pendragon08/humanoid-combat-motion.git
cd humanoid-combat-motion
```

不同实验使用独立环境。先选择要复现的流程，再准备对应依赖：

| 流程 | 环境入口 | 用途 |
|---|---|---|
| 单人扩散先导实验 | [env-spec.json](env-spec.json) | Python 3.10、PyTorch 2.5.1 / CUDA 12.1、MuJoCo 3.3.7、OSQP 1.0.5 |
| 双人生成与适配 | [TIMotion 环境](docs/timotion_environment.md) | InterHuman 表示、模型推理与适配 |
| 人体到机器人重定向 | [重定向环境](docs/retarget_environment.md) | 人体骨架到 G1 参考动作 |
| 学习型动作跟踪 | [PBHC 复现](docs/kungfubot_official_reproduction_20260917.md) | Isaac Gym 下的单动作控制基线 |
| EngineAI 平台探索 | [官方环境记录](docs/engineai_official_environment.md) | 众擎仿真与单机器人验证入口 |

环境文件记录的是实验依赖，不是完整可分发环境镜像。旧安装脚本依赖原服务器路径、共享解释器及磁盘审计文件，不能在新机器上直接当作通用安装器。单人流程需先建立仓库根目录 `.venv`，并按 `env-spec.json` 准备依赖；后续命令通过 `scripts/run.sh` 使用该环境。

基础检查：

```bash
bash scripts/run.sh scripts/smoke.py
```

## Phase 1: Data Preparation

单人先导实验使用 [KungFuAthleteBot](https://github.com/NPCLEI/KungFuAthleteBot) 的视频恢复及 G1 重定向动作。它们是视频来源参考动作，不能标作专业光学 MoCap 真值。

在确认数据许可与下载位置后：

```bash
bash scripts/run.sh scripts/download_resources.py
bash scripts/run.sh scripts/prepare_data.py
```

下载脚本固定数据与机器人资产版本，预处理输出为 `data/windows.npz`。原始片段、切片和派生动作按来源分组，避免同源片段跨训练与测试集合。

双人流程与单人关节角流程分别处理：TIMotion 使用人体交互表示，不能直接作为 G1 关节角。相关任务、数据划分和对照约定见 [基准协议](docs/benchmark_protocol_20260915.md)、[ReMoCap 实验](docs/remocap_execution_20260916.md) 与 [实施说明](docs/implementation_audit.md)。

## Phase 2: Motion Generation

单人先导模型是 6 层时序卷积去噪器，用于种子动作条件增广。环境及数据准备完成后，在新的输出目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh scripts/train.py \
  --seed 0 --steps 3000 --run pilot_reproduction_seed0

bash scripts/run.sh scripts/evaluate.py --run pilot_reproduction_seed0
bash scripts/run.sh scripts/render_results.py --run pilot_reproduction_seed0
```

这里训练的是动作生成模型，不是 RL 控制器。关节范围投影、地面几何修正及简单 PD 回放属于先导诊断，不能替代学习型跟踪与动力学可执行性验证。

双人条件模型、攻击—反应适配与接触修正的入口见 [双人生成记录](docs/paired_execution_progress.md) 和 [双人精化记录](docs/paired_refinement_execution.md)。人体接触代理、机器人实际接触和任务成功分别评价。

## Phase 3: Tracking and Evaluation

研究重点是生成数据是否改善学习。因此，跟踪算法与数据配置需要分开控制：

1. 在固定机器人与跟踪器下检查参考动作的可执行性。
2. 在相同训练预算下比较已有数据、普通生成增广与接触适配增广。
3. 报告来源划分、动作数、种子数、提前终止及失败案例。

三组数据的历史协议见 [训练价值对比](docs/step7_protocol.md)。PBHC 单动作原方法复现见 [复现说明](docs/kungfubot_official_reproduction_20260917.md)；EngineAI 分支见 [单机器人实验](docs/engineai_single_experiment.md)。

新的仿真 RL 训练须遵循 [训练可视化要求](docs/training_visualization_policy_20260917.md)。历史配置和运行目录保留，不因 README 更新而重启实验。

## Results

| 实验 | 已归档结果 | 能够支持的结论 |
|---|---|---|
| PBHC 官方单动作基线 | 50,000 次更新；评价回合长度比约 99.35% | 单动作跟踪基线可运行，不是合成数据收益 |
| TIMotion 原模型 / 条件适配器 | FID 5.5111 / 5.5344；Top-1 49.12% / 48.53% | 1099 条可用测试样本上，未观察到整体适配收益 |
| 数据训练价值对比 | 严格通过率：视频来源 3.39%、普通增广 0.78%、接触适配增广 0.78% | 三种子、有限开发来源上的负面结果 |
| 双人几何修正 | 穿插帧减少，但两组机器人严格成功均为 0/48 | 几何改善未转化为动态执行成功 |

PBHC 指标是 **episode length ratio（回合长度比）**，不是任务成功率。历史数据对比的留出动作仅来自一个视频标题组，也不能作为最终独立泛化结论。

证据入口：[PBHC 结果](reports/PBHC_OFFICIAL_TRAINING_RESULTS_20260918.html)、[BM1 结果](records/bm1_20260915_v1/suite_result.md)、[数据效用指标](records/step7/summary.json)、[结果说明](docs/results_summary.md)。

## Repository Structure

```text
combat/        动作生成、约束、接触代理及机器人接口
configs/       数据划分、实验协议与平台配置
scripts/       数据准备、生成、重定向、训练及评价
tests/         表示、接触代理与数据协议检查
records/       原始指标、版本清单及执行凭据
docs/          实验说明、环境与方法边界
reports/       HTML 展示、结果分析与视频入口
```

## Documentation

- [结果概览](docs/results_summary.md) · [实验记录](docs/experiment_log.md)
- [数据训练价值协议](docs/step7_protocol.md)
- [跟踪器升级与算法说明](docs/tracker_upgrade_and_algorithm_20260916.md)
- [剩余实验与恢复条件](docs/remaining_steps_20260916.md)
- [整理前的完整 README](https://github.com/Pendragon08/humanoid-combat-motion/blob/8e351789fe672214d8c31f47eceac78181e3dda2/README.md)

HTML 报告可下载后在浏览器中打开。视频中的参考回放、固定策略评价与训练场景按各报告说明区分。

## Acknowledgements and License

感谢 [KungFuAthleteBot](https://github.com/NPCLEI/KungFuAthleteBot)、[TIMotion](https://github.com/AIGC-Explorer/TIMotion)、[InterGen / InterHuman](https://github.com/tr3e/InterGen)、[GMR](https://github.com/YanjieZe/GMR) 及相关控制框架提供的数据、生成与机器人基础。具体采用版本和接口差异见 [实施说明](docs/implementation_audit.md) 与各实验环境记录。

仓库目前未提供统一的顶层 LICENSE。上游数据、模型和机器人资产遵循各自许可；公开代码不表示所有数据与权重均可再分发。引用实验时请注明源数据、跟踪器、提交版本和评价协议，并引用对应上游方法。
