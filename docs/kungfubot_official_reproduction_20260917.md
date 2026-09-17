# 修订：先建立KungfuBot官方基线，再验证示范数据创新

## 本轮决定

优先复现KungfuBot第一篇的官方单动作训练流程。此前BeyondMimic适配、固定策略及质量修正实验保留为开发诊断，不能代称KungfuBot论文复现。当前已经启动的四组等预算对照按其有限队列完成和归档，不再自动扩大数据修正试验。

研究对象仍是低成本、高质量格斗示范数据构建。复现PBHC是为建立可信的下游验证基线，不把官方的控制器、PPO、课程机制作为我们的创新。

## 发现并纠正的关键差异

|项目|此前适配实验|本轮官方复现|
|---|---|---|
|实现|BeyondMimic官方框架与我们的包装脚本|TeleHuman/PBHC官方单动作实现|
|任务配置|Tracking-Flat-G1-v0|exp=motion_tracking；MHPPO|
|仿真|IsaacLab / IsaacSim|IsaacGym Preview 4|
|机器人|29关节参考，补中性腕关节|原g1_23dof_lock_wrist|
|输入|转换后的50Hz NPZ|官方原始Horse-stance_punch.pkl|
|控制和奖励|BeyondMimic配置|官方非对称策略/多头价值网络、adaptive tracking sigma（自适应跟踪容差）、课程和随机化|
|单动作预算|1024×24×2000步；新诊断512×24×1000步|4096×24×50000=4,915,200,000环境步|
|主要评价|自定义全程逐帧严格判据|官方sample_eps.py的误差/平滑性，ratio_eps.py的ELR（执行时长比例）|
|旧0%结果|保留|单独列为自定义严格评价，不能冒充官方指标|

官方仓库训练说明要求4096环境、50000更新，但配置文件默认最大迭代数为1000000；启动命令必须显式覆盖为50000，避免无界训练。前一轮1024环境×2000更新的单动作环境步只有该预算的1/100，且算法、模型均不同；不能据此判定原论文失败，也不能假设补足预算必然成功。依据：[固定版本官方训练说明](https://github.com/TeleHuman/PBHC/blob/ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0/humanoidverse/README.md)。

## 复现边界与分阶段门槛

1. **已完成的只有部署样例复现**：官方马步、马步冲拳ONNX权重在MuJoCo运行；不是从零训练复现。
2. **环境与短测试**：独立Python3.8环境、IsaacGym物理/梯度验证、128环境两更新、4096环境两更新容量检查。检查点和真实环境步均留证据，不能只看进程退出。
3. **官方权重锚点**：直接使用提供的model_33000.pt执行官方两类评价，预检查各100个episode（完整回合）。先确认评价链可用。
4. **独立复核**：新上下文代理照下面命令原样执行物理验证，确认环境文档和实际一致后，才开放完整训练。
5. **第一条正式训练**：原始马步冲拳，随机初始化seed1，4096环境、50000更新，保留全部官方学习组件。没有数据修正或挑选“最好”检查点。
6. **同口径评价**：官方提供权重与新训练权重各按原脚本评价1000回合。原文采用3个训练种子；当前只启动1条动作、1个种子，因此仍是训练阶段复现的第一步，不是完整论文结果表。只有基线可信后再考虑3种子、多动作和数据创新对照。[原文§5.1、附录D](https://arxiv.org/html/2506.12851v1#S5.SS1)。

没有自动排入实机、视频到SMPL全流程或所有消融。这些尚不具备已验证的完整入口。本轮也不借减少训练随机化、缩短参考或放宽旧指标制造成功。

## 评价口径

官方sample_eps.py禁用早停及大部分随机化，但保留控制延迟；ratio_eps.py保留早停并设置其自己的motion-far阈值。两种成绩分开报告。原误差使用毫米、关节向量范数及逐帧差分单位；不能与此前关节MAE、米/秒统计直接拼成同一表。

官方六类主要误差为全局/相对根的身体位置、关节位置/速度、身体速度/加速度，另报告原实现平滑性及ELR。历史严格逐帧通过率仍保留为补充；本轮没有对其阈值做修改。

发现上游保存命名使用最近一次周期保存的iteration字段，循环结束会覆盖该名称。例如50000更新完成时可能仍名为model_49000.pt。为此仅增加只读计数/保存观察包装，记录真实环境步和检查点SHA，不修改PPO、奖励或采样；不能从文件名推断更新次数。Python/NumPy/Torch显式种子由启动器设定并记录。

## 固定来源和工程差异

- PBHC提交：ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0。官方仓库包含KungfuBot2，但本轮不调用general_tracking或teacher/student配置。
- 环境：env-pbhc-training.json；Python3.8.20、Torch1.13.1+cu117、NumPy1.24.4、IsaacGym Preview4。Torch及依赖版本是此次工程固定值，不冒称论文全部精确版本。
- NVIDIA官方发布包已可下载；不使用IsaacLab替代IsaacGym。已有训练环境保持独立。
- 初次安装解析器试图升级Torch，已在安装完成前停止该步骤，加入Torch/torchvision约束后重试；日志保留。没有改动官方受版本控制代码。
- 2026-09-17预检查首次失败于Ninja解析：固定版本Ninja已安装在独立环境中，但启动器只调用了该环境Python，没有把其bin加入PATH。修复仅在启动器前置环境bin，并更新环境规格；不重装仿真、不修改官方算法。原失败日志归档后进行一次有界预检查重试，通过后仍须新的独立见证。
- 当前环境安装/验证和训练的真实状态，分别见records/pbhc_official_reproduction_20260917下bootstrap_state、preflight_state、train_state。缺少TRAINING_COMPLETE时不得写“训练已完成”。

## 后续示范数据实验

原始PBHC基线达到可解释水平后，再固定这一官方实现，比较原始数据与我们的修正/扩增数据。各组独立训练、同预算/种子/机器人/指标；与跨方法组件对照分开。创新仍须由数据来源成本、动作覆盖、接触质量及同一基线下的训练收益支撑，不能用借来的控制器性能冒充数据创新。

## 独立环境验证命令

先确认preflight已完成、两张卡均空闲；以下从本地PowerShell执行一次。不要编辑、安装、重试或调整命令；失败直接记录分歧。输出路径必须原先不存在。

```powershell
ssh -b 10.147.17.250 -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_pbhc_training.sh /data/code/ouyanghao/oyh/humanoid-combat-motion/scripts/witness_pbhc_gym.py --gpu 0 --output /data/code/ouyanghao/oyh/humanoid-combat-motion/records/pbhc_official_reproduction_20260917/gym_independent_witness.json'
```

必须退出0，输出PBHC_GYM_PHYSICS_WITNESS，4个物体完成120物理步、最终高度0.05–0.2m、梯度有限。该证据只验证环境，不能代表动作成功。复核代理记录agent、status、invocation_unmodified和环境规格canonical SHA前8位到independent_witness.json；调度器缺该记录时拒绝正式训练。
