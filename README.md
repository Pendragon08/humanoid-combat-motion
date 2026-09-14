# 接触与物理约束格斗示范数据生成

## 当前进度（2026-09-14）

根据上传实施方案，在原 pilot 基础上增量实现双人路线。详细状态见 [实施审计](docs/implementation_audit.md)、[实验日志](docs/experiment_log.md) 和 [结果状态](docs/results_summary.md)。
当前已新增接触评价、接触优化、结构化条件适配器、轻量物理评价网络和数据划分检查。官方 TIMotion Transformer CUDA 模块测试已通过；完整预训练复现、Isaac Lab 跟踪与三种子数据效用比较仍在执行准备中，**不能宣称方案已经验证成功**。

下面保留原先单人 pilot 的复现说明，其结果不等于双人格斗或学习型跟踪实验。

## 旧版 G1 pilot

这是博士课题的第一版可运行代码。它以公开武术视频恢复并重定向得到的 G1 动作为起点，训练小型动作扩增模型，进行关节约束修正、地面接触几何修正和独立的自由浮动机器人仿真诊断。

**本轮没有训练机器人模仿策略，不能据此宣称合成数据已经提高机器人学习能力。** 双人攻击—防御、击打接触力、文字指令生成和实机部署也尚未实现。这里首先检验：生成的参考动作在修正后，是否减少明确可测的约束违规，以及是否能被一个简单控制器跟踪。

## 术语与实现边界

|术语|简单解释|本轮状态|
|---|---|---|
|Reference Motion（参考动作）|供控制器跟随的目标姿态与时间序列|已实现数据接口|
|Motion-to-Motion Diffusion（动作到动作的扩散扩增）|先给已有动作加噪，再通过训练好的网络生成变体|小型 6 层时间卷积网络，预测干净关节角|
|Root（根部位姿）|机器人骨盆在世界中的位置与朝向|作为输入条件沿用种子动作，没有从零生成|
|DOF（自由度）|可独立运动的关节数量|G1 的 29 个关节|
|Retargeting（动作重定向）|把人体动作映射为机器人的动作|复用上游 GMR 已处理的数据，显式核对关节顺序|
|QP（凸二次规划）|在尽量少改变原动作的前提下满足一组约束|固定时间，不靠放慢播放消除速度违规|
|Kinematic Feasibility（运动学可行性）|关节角、速度和加速度是否符合设定范围|可检查，不能替代动力学验证|
|Dynamic Feasibility（动力学可行性）|考虑重力、质量、接触和电机后能否真正执行|通过独立仿真诊断；简单 PD 失败不证明所有控制器都失败|
|PD Controller（比例—微分控制器）|按位置误差与速度误差计算电机力矩|有力矩限制、自由浮动根部，无额外扶稳力，不是学习得到的策略|
|Ground Contact（地面接触）|足部是否接近地面以及是否滑动|根据种子几何推测；不是测得的接触力|
|Combat Contact（格斗接触）|拳脚与对方身体或防御肢体的接触关系|尚未实现|

## 数据和许可证

- [KungFuAthleteBot](https://github.com/NPCLEI/KungFuAthleteBot) / [数据集](https://huggingface.co/datasets/LuluCao/KungfuAthleteBot)：源自武术视频，经 GVHMR、GMR 与人工筛选；**不能标成真实光学 MoCap**。
- 下载器固定版本 `ff09f3795dc99bef73b21413b03fa8470917a22c`，按源视频标题筛出地面长拳、南拳、太极拳；不下载全库。实际帧率以 NPZ 字段为准。当前样本实际为 50Hz，统一重采样到 30Hz，四元数采用球面插值（SLERP）。
- [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco) G1 模型固定版本 `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`。
- [GMR](https://github.com/YanjieZe/GMR) 的关节顺序和模型限制固定版本 `bb1bbe40774794fceb2a7c579a3464a28e68c844`。
- 下载记录、各文件 SHA256（数字校验值）及原始许可证保存于 `data/manifest.json` 与 `assets/*LICENSE`。数据卡的 Apache-2.0 标记不能自动替代源视频与人体模型的独立授权条款。

按原视频标题分组：长拳三段片段留作测试，长拳五段片段留作验证，其余用于训练。每条长片最多取 48 个窗口，窗口 64 帧。没有根据测试结果重新挑选训练数据。演员身份未知，因此不是严格的跨演员基准。

所有比较组都先使用同样的初始高度对齐：根据 Unitree 模型的足部碰撞点，仅给整个窗口施加一个固定竖直偏移，使初始最低足部碰撞点距地面 2mm，偏移最多 12cm。这样避免因源模型与仿真模型的足部几何差异，造成某些组一开始就穿地；并未对原动作逐帧吸附地面。每个窗口的偏移量均保存。

## 环境

Python 3.10、PyTorch 2.5.1+cu121、MuJoCo 3.3.7、OSQP 1.0.5。服务器使用独立 `.venv`，只读复用已有 PyTorch 库，所有写入和缓存位于本项目。安装前声明见 `env-spec.json`。

已配置服务器目录：`/data/code/ouyanghao/oyh/humanoid-combat-motion`。本机代码目录：`D:\具身智能\humanoid-combat-motion`。

```bash
cd /data/code/ouyanghao/oyh/humanoid-combat-motion
# 先验证实际 GPU 运算、物理步进和画面渲染
bash scripts/run.sh scripts/smoke.py
# 网络可用时下载；当前服务器不能直连 HF，因此由本机下载后传入
bash scripts/run.sh scripts/download_resources.py
bash scripts/run.sh scripts/prepare_data.py
```

迁移到其他服务器时，先修改 `env-spec.json` 中的解释器路径，或者新建完整环境；不要直接照搬旧服务器绝对路径。

## 可复现实验命令

以下是实验命令，是否已经执行完成以 `runs/*/TRAINING_COMPLETE.json` 与 `EVALUATION_COMPLETE.json` 为准。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh scripts/train.py --seed 0 --steps 3000 --run pilot_seed0
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh scripts/train.py --seed 1 --steps 3000 --run pilot_seed1
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh scripts/evaluate.py --run pilot_seed0
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh scripts/evaluate.py --run pilot_seed1
bash scripts/run.sh scripts/render_results.py --run pilot_seed0
bash scripts/run.sh scripts/export_reference.py \
  --input runs/pilot_seed0/motions.npz --case 0 --arm 3 \
  --output runs/pilot_seed0/example_tracking_reference.npz
```

测试样本的选取规则在结果视频中固定：每个测试源片段的第一个评估窗口，不能只展示成功案例。

## 五组比较

1. `video_seed`：视频恢复并重定向后的原参考动作。
2. `generic_diffusion`：相同种子条件下的扩散生成变体。
3. `joint_constrained`：对第 2 组施加关节位置、速度和加速度约束。
4. `ground_refined`：进一步修正根部位置，尽量恢复种子动作的足部接触几何关系。
5. `physics_selected`：在噪声等级 20、40、60 下分别生成并修正候选，实际仿真后按事先固定的代价函数选择一个。每次多使用三次候选仿真，保存全部候选得分与失败，不把它作为等计算量比较。

**第 4 组仍然不是完整的“动力学约束生成”或“双人格斗接触生成”。** 最后另外执行实际 MuJoCo 自由浮动仿真，把不能被当前简单控制器执行的动作标记为失败。失败样本保留在结果中。

第 5 组是有限预算的 Physics-in-the-loop（物理仿真参与筛选）实现。固定代价为根部跟踪误差 + 0.2×关节跟踪误差 + 2×摔倒后的剩余时间比例 + 0.1×偏离种子动作的关节误差。筛选和最后诊断使用同一控制器，因此不能宣称对未见控制器的泛化；如果所有候选均失败，最优候选仍标为失败。

摔倒判据：骨盆低于 0.35m，或根部朝上轴与世界竖直轴夹角的余弦低于 0.25。完整跟踪还要求关节均方根误差低于 0.35rad、根部均方根误差低于 0.25m，且不存在数值异常。这些都是初期诊断阈值，不等同于人工标注的格斗任务成功。

关节角与力矩限制来自固定的 Unitree 模型；速度上限来自固定的 GMR G1 URDF；加速度上限设为 100 rad/s²，仅是本轮研究筛选参数，不能声称为厂家规格。质心支撑区检查只是准静态近似，不是高速踢击的充分稳定性条件。

## 输出

- `audit/`：磁盘清理、环境、来源与关节顺序审计。
- `runs/*/loss.jsonl`：原始训练与验证损失，不抹除失败。
- `runs/*/best.pt` / `last.pt`：验证集选出的模型和最终模型。
- `runs/*/motions.npz`：所有测试参考动作，含关节名称与四元数约定。
- `runs/*/metrics.json`：逐样本指标和物理失败记录。
- `runs/*/media/reference_comparison.mp4`：**参考动作回放，不是动力学执行视频**。
- `runs/*/media/physical_pd_diagnostic.mp4`：实际仿真状态的回放，显式显示摔倒。
- `example_tracking_reference.npz`：未来接入跟踪框架的中间数据格式。必须按目标框架检查关节/身体名称顺序后使用；尚未宣称与 BeyondMimic 即插即用。

## 下一步如何真正检验课题

把本轮参考数据接入同一套 G1 跟踪框架，在相同训练预算下比较：真实或视频种子数据、种子加普通生成数据、种子加约束生成数据。采用多个随机种子、固定未见动作测试集，报告跟踪误差、摔倒率、泛化以及合成数据数量与性能的关系。只有这一阶段才能回答“合成动作是否改善机器人模仿学习”。

随后加入攻击—防御角色、接触部位与接触时刻标签，并用真实跟踪策略的执行反馈改进生成器。当前代码是该路线的起点，不是最终论文方法的完整复现。
