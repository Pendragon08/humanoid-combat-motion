# 众擎官方指定版本验证环境

本环境用于排除上一轮 Isaac Lab 2.1 接口适配的不确定性。环境规格为 `configs/engineai_official_env.json`。保留旧环境、检查点和失败记录。

采用众擎 README 指定的 Isaac Lab 提交 `c22775241e28f465fe345fa1a482ad6d29d712b0`，而不是仅根据 v2.3.2 标签选择另一提交；众擎源码固定 `14ec57be718586bd0ac45375aa1115bd896fbdbc`。该 Lab 提交的官方兼容表包括 Isaac Sim 4.5，因此只读复用服务器原有 standalone 4.5.0；不声称作者使用了同一 Sim 小版本。独立 Python 3.10 环境安装 Torch 2.7.0+cu128 和该提交要求的 RSL-RL 5.0.1，直接使用原生 Actor/Critic 模型及官方 PM01 任务。没有上一轮的配置导入屏蔽和四元数兼容补丁。

平台来源：[EngineAI](https://github.com/engineai-robotics/engineai_rl_lab)、[固定 Lab 提交及兼容表](https://github.com/isaac-sim/IsaacLab/blob/c22775241e28f465fe345fa1a482ad6d29d712b0/README.md)。官方指定提交与 Git 标签不一定相同，本实验以其明确指定提交为准。

补充依赖锁定：Pinocchio2.7.0、hpp-fcl2.4.4满足官方的最低版本约束且兼容NumPy1.26；最新Pinocchio4依赖的Boost轮子要求NumPy2，与Lab的numpy<2冲突。wheel0.45.1用于兼容Lab的packaging<24。这些是宿主机依赖解析选择，不能说作者提供了这一完整锁文件；最终pip freeze保存所有解析版本。

运行时使用Isaac Sim自带Python3.10解释器和标准库，独立venv只提供新依赖包，通过入口处理`.pth`文件。首次使用conda系解释器搭配Kit标准库导致`platform`版本字符串解析失败，已保存日志；没有修改Python标准库来掩盖该错误。Eigenpy锁定3.5.1以匹配Boost1.83。

入口先解析隔离环境的Torch/Torchvision/NumPy/Gymnasium，再启动Kit。Kit快速导入器会把Torch2.7的子模块指向自带2.5目录，因此为明确列出的数值包使用标准Python PathFinder，并保留Kit对仿真命名空间的导入机制。曾尝试全局关闭快速导入器，但导致isaacsim命名空间无法解析，已保留失败日志并取消该设置。加载隔离不改变物理、奖励或动作参考；版本及已加载Torch模块路径断言禁止混用2.5。

## 磁盘与安装

2026-09-16 安装前 `/data` 可用约142GiB，系统盘约15GiB。全部新增包、源码、临时文件和实验结果放到项目的数据盘路径；安装预检要求至少60GiB剩余，不删除原数据。没有安装新的完整 Isaac Sim，没有升级服务器系统或GPU驱动。

源码通过官方 codeload 的固定提交压缩包获得，记录压缩包SHA256。Git拉取连接卡住的首次尝试保留日志，换官方压缩包获取不改变代码版本。安装命令为 `python3 scripts/setup_engineai_official.py`，日志为 `records/engineai_official_v1/build_archive.log`。`INSTALL_COMPLETE.json` 只代表包安装完成，不代表仿真通过。

首次服务器默认镜像下载较慢且自动解析选中NumPy2依赖；锁定后保留完整wheel文件，使用 `scripts/install_engineai_cached.py` 离线解析并从官方PyPI仅补取缺包，记录在 `build_cached.log`。脚本支持保存全部依赖冻结清单。首次安装仍需联网，不能把本机缓存作为公开可复现的前置条件。

## 独立核验（只运行，不能修改环境）

从当前电脑 PowerShell 原样执行：

```powershell
ssh -b 10.147.17.250 -o ConnectTimeout=10 -o BatchMode=yes -i C:\Users\Pende\.ssh\id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_engineai_official.sh scripts/engineai_official_validate.py --mode witness --motion third_party/engineai_rl_lab/datasets/tracking/pm01/dance.npz --checkpoint cache/engineai_overlay/dance.pt --output runs/engineai_official_v1/witness_independent_v2 --num_envs 4 --headless --device cuda:1"
```

输出应有 `ENGINEAI_OFFICIAL_WITNESS`，退出0，产生 `protocol.json`、`results.json`、`rollout_env0.npz`；200个控制步对应每环境800个物理步，有限CUDA反向传播损失。协议必须显示Torch2.7.0+cu128、RSL-RL5.0.1、从新路径加载Lab。实际调用原生RSL-RL模型，直接加载官方actor/critic，不手工转换网络结构。见证通过只证明环境和策略接口可运行，不代表全部动作成功或数据有效。若输出目录已存在，不覆盖、不更名重跑，报告已有状态。

之后完整官方示例评估固定起点、关闭观测噪声、把评估时限延长到参考结束后；保留官方动力学随机化、奖励和物理终止条件。无外部视觉材质和调试标记仅改变显示。所有这些评估差异均记录，不称为原封不动的官方 play.py 运行。两类生成数据随后使用同一评价规则；如微调，则同预算100轮、256环境、614400环境步、同官方先验、重置优化器和探索标准差0.3。它仍然是参考拟合试验，不是独立测试集的数据增益实验。

版本记录补充：Kit环境中的distribution metadata可能首先找到Sim自带Torch2.5元数据，即使实际加载的是隔离环境Torch2.7。协议现在分别记录实际模块版本、模块路径与distribution metadata，不覆盖旧见证记录。实际Torch及其已加载子模块必须通过2.7和隔离路径断言。
