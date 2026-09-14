# Implementation audit / 实施审计

2026-09-14；用户上传方案的增量实现，不另起一个脱离原项目的代码库。

## Existing implementation

原项目包括 KungFuAthleteBot 单人视频重建动作、G1 29 关节扩散小模型、关节/速度/加速度投影、地面几何修正、MuJoCo 简单 PD（按位置误差反馈的控制器）诊断和两个随机种子。
不包括 TIMotion、双人条件模型、接触模型、Isaac Lab 或学习得到的跟踪策略。原 pilot 全部严格筛选通过数为零，不能作为本次下游学习有效性的证据。旧文件和结果保留。

## Official source audit

| Component | Verified source | Finding |
|---|---|---|
| TIMotion | https://github.com/AIGC-Explorer/TIMotion/tree/172580fd7f2551a7cc10fde1f15ff73315742e65 | models/blocks.py 的 RWKV 导入被注释；实际为 TIMotionTransformerBlock；选择计划允许的 Transformer 路线，不能称 RWKV 复现 |
| InterHuman | https://github.com/tr3e/InterGen | 公开 Drive 提供 motions_processed.zip，2,462,028,219 字节；需另取文本和原始划分；数据不提交 GitHub |
| BeyondMimic | https://github.com/HybridRobotics/whole_body_tracking/tree/cd65172032893724b445448818c34165846d847d | 官方要求 Isaac Lab 2.1.0；当前服务器未安装 |
| Isaac Lab | https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html | Ubuntu 20.04 有二进制安装路线，不能因为 pip 的 GLIBC 限制就断言完全不可运行；尚未完成安装与物理验证 |

TIMotion 真实表示为每人 262 维，必须沿用官方预处理和归一化；其骨架位置字段不能当作 G1 关节角。后续 retargeting（把人的动作映射到机器人）必须保持双人相对位置，不能分别归零根位置。

## New implementation

- `combat/interaction.py`：共享世界坐标下 Hit / Block / Avoid 球体接触代理；线性帧间扫掠检测快速穿透；保留伪标签置信度和来源。
- `combat/protocol.py`：源序列不交叉检查，禁止测试条件生成训练样本；物理标签必须来自有检查点和轨迹校验值的学习型控制器，拒绝把 PD 诊断充当标签。
- `configs/paired_experiment.json`：三种子、五组数据、公平预算、消融和正式评价验收条件。
- `tests/test_interaction.py`：命中、未命中、格挡失败、闪避、穿透、坐标平移一致性和数据泄漏测试。

这些模块是实现进展；测试通过不代表 TIMotion 复现成功或博士方案已经实证有效。

## Resources

服务器：Ubuntu 20.04.6，GLIBC 2.31，驱动 570.181，2×A6000 48GB，约 247GiB 可用内存。根盘约 15GB 空闲，数据盘约 43GB 空闲；新增下载和环境必须控制容量。旧 TD-MPC2 已归档和删除，本次不删除其他研究项目。

本地 D 盘仅约 3GB 空闲，官方大文件暂存到本地 C 盘的本任务 work 目录，校验后上传服务器。下载不完整文件用 `.partial` 标记，不能被训练读取。
