# 人形机器人格斗示范数据生成算法实施报告（Codex 执行版）

## 当前范围与执行入口（2026-09-15修订）

**研究重点是构建可补充或减少专业真人MoCap用量的高质量格斗示范数据。Motion Tracking（动作跟踪）是统一的下游验证工具，不是论文算法贡献。**

当前协议以 [第七步协议顶部](step7_protocol.md) 和 `configs/data_utility_protocol_v2.json` 为准：来源与成本审计 → 格斗条件生成 → 接触/物理约束 → 参考数据 → 固定学习方法分别训练 → 独立评估和MoCap剂量曲线。现有KungFuAthlete种子属于视频恢复，不能作为传感器MoCap基线。

下文保留原上传方案及技术候选供追溯，其中“三个创新模块”、完整物理critic、跟踪器开发顺序均不视为已验证创新或必须实现的当前任务。已有控制器和重定向实验归入工程诊断；不继续将控制算法改进作为首篇核心。新协议尚未启动训练。

---

## 0. 文档目的

本文档用于指导 Codex 在现有项目基础上，实现一个可实验、可消融、可复现的“面向人形机器人模仿学习的格斗示范数据生成”方案。

当前暂定课题：

> **面向人形机器人模仿学习的接触与物理约束格斗示范数据生成方法**

英文暂定：

> **Contact- and Physics-Constrained Combat Demonstration Generation for Humanoid Robot Imitation Learning**

本方案不以“从零训练一个大模型”为目标，而是在已有双人动作生成模型和物理仿真工具基础上，针对人形机器人格斗场景增加三个明确创新模块：

1. **Combat-conditioned Causal Reaction Generation**  
   面向攻击—防御—反击关系的格斗因果反应生成。
2. **Combat Interaction Contact Guidance**  
   面向 Hit / Block / Avoid 的时空接触约束生成。
3. **Dual-Robot Combat Physics Critic**  
   面向双机器人格斗的物理可执行性评价与筛选。

最终必须证明：

\[
D_{\mathrm{real}} + D_{\mathrm{ours}}
\]

相比

\[
D_{\mathrm{real}}
\]

和

\[
D_{\mathrm{real}} + D_{\mathrm{generic\ synthetic}}
\]

能够真正提升下游 humanoid motion tracking / imitation learning 性能。

---

# 1. 执行总原则

Codex 在开始修改任何代码之前，必须先扫描真实项目。

不要根据本报告假设项目已经具有某个目录、类、接口或训练脚本。

必须优先完成：

1. 扫描项目目录树；
2. 找到现有 motion generation / dataset / training / evaluation / simulator 相关代码；
3. 找到已有配置文件；
4. 找到已有模型权重加载逻辑；
5. 找到已有日志与实验输出规范；
6. 判断当前项目是否已经集成 TIMotion / InterGen / Isaac Lab / retargeting；
7. 在现有工程结构上增量实现，避免重写已有功能。

禁止：

- 未扫描项目就大规模重构；
- 为了“看起来完整”重复实现已有工具；
- 修改现有数据划分而不记录；
- 把 Human Motion、Robot Reference Motion、Motion Tracking 混为一个概念；
- 把 RL rollout 数据当作本课题的 demonstration data；
- 为了跑通流程而跳过 contact / physics 评价。

---

# 2. 科研问题定义

本课题不是单纯研究：

\[
Text \rightarrow Motion
\]

而是研究：

> 如何在减少专业真人 Motion Capture 依赖的情况下，低成本、可扩展地生成具有正确格斗语义、攻击—防御关系、接触关系，并满足目标人形机器人动力学可执行性的示范数据。

完整目标：

\[
\text{Combat Condition}
\]

↓

\[
\text{Two-Person Motion Generation}
\]

↓

\[
\text{Combat Contact Constraint}
\]

↓

\[
\text{Robot Physics Evaluation}
\]

↓

\[
\text{Robot Reference Motion}
\]

↓

\[
\text{Motion Tracking / Imitation Learning}
\]

↓

\[
\text{Downstream Performance Evaluation}
\]

---

# 3. 当前暂定基础模型

## 3.1 主 Backbone

第一版主模型暂定：

\[
\boxed{\text{TIMotion-RWKV}}
\]

理由：

- 任务本身就是双人 human-human motion generation；
- 已包含 temporal interaction 建模；
- 已包含 causal interaction 思路；
- 已考虑 role evolution；
- 适合作为 Attacker / Defender 双分支基础；
- 不需要从零训练大型 Diffusion Transformer；
- 更符合 2 × RTX A6000 的算力条件。

若当前项目中 TIMotion-RWKV 无法稳定复现，则按以下优先级降级：

1. TIMotion-RWKV；
2. TIMotion-Transformer；
3. InterGen；
4. 其他当前项目已稳定复现的 two-person diffusion baseline。

**不要同时维护多个主模型。**

第一阶段只选一个主 Backbone 完成全流程，其余作为 baseline。

---

## 3.2 基线模型

建议至少包含：

### B0：真实数据基线
仅真实/现有示范数据，不增加 synthetic motion。

### B1：InterGen
普通双人 interaction generation。

### B2：TIMotion
原始 TIMotion，不加入格斗专用模块。

### B3：TIMotion + Combat Fine-tuning
只使用格斗数据微调，不加入 Contact 和 Physics Critic。

### B4：TIMotion + Combat + Contact
加入 Innovation 1 和 Innovation 2，不加入 Physics Critic。

### B5：Ours
完整：

\[
Combat\ Causality
+
Combat\ Contact
+
Robot\ Physics
\]

---

# 4. 数据表示

## 4.1 Human Motion 表示

优先沿用 TIMotion / InterHuman 现有 motion representation。

若使用 TIMotion 原格式，建议保持：

- 22-joint kinematic structure；
- 6D joint rotation；
- root translation；
- root orientation；
- frame rate 与原数据集一致；
- 统一 canonical orientation。

不要第一版就修改底层 motion representation。

---

## 4.2 两个人的动作

每个样本：

\[
X =
\{
M_A,
M_B,
C_{\mathrm{text}},
C_{\mathrm{combat}},
C_{\mathrm{contact}}
\}
\]

其中：

### \(M_A\)
Attacker / Actor motion。

### \(M_B\)
Defender / Reactor motion。

### \(C_{\mathrm{text}}\)
自然语言描述。

### \(C_{\mathrm{combat}}\)
结构化格斗条件。

### \(C_{\mathrm{contact}}\)
时空接触约束。

---

# 5. Combat Condition Representation

新增一个结构化格斗条件：

```text
CombatCondition:
    attack_type
    attack_limb
    attack_target
    defense_type
    defense_limb
    counterattack_type
    actor_role
    reactor_role
    timing(optional)
```

例如：

```yaml
attack_type: right_jab
attack_limb: right_hand
attack_target: head
defense_type: left_forearm_block
defense_limb: left_forearm
counterattack_type: right_cross
actor_role: attacker
reactor_role: defender
```

第一版不要求所有字段都有人工标签。

可以先支持：

- attack_type
- attack_target
- defense_type
- role

后续再扩展 timing 和 counterattack。

---

# 6. 创新点 1：Combat-conditioned Causal Reaction Generation

## 6.1 要解决的问题

现有 two-person motion generation 能生成“两个人一起运动”，但未必形成正确的格斗逻辑：

\[
Attack
\rightarrow
Defense
\rightarrow
Counterattack
\]

例如：

- 防御动作时机错误；
- 防御动作和攻击类型无关；
- 攻击方向变化后防御仍保持相同动作；
- 角色切换不自然。

---

## 6.2 方法依据

主要参考：

- TIMotion：Causal Interactive Injection、Role-Evolving Scanning；
- HERO：reaction generation；
- Human-X：reaction diffusion + physics-aware interaction。

我们的贡献不是“首次提出 causal interaction”，而是：

> 将通用 human-human causal interaction 改造成结构化的 combat-conditioned reaction generation。

---

## 6.3 实现方案

在 TIMotion Backbone 前增加：

\[
Combat\ Condition\ Encoder
\]

将：

\[
C_{\mathrm{combat}}
\]

编码为 condition embedding：

\[
e_c = f_{\mathrm{combat}}(C_{\mathrm{combat}})
\]

Attacker 分支：

\[
M_A = G_A(z, e_{\mathrm{text}}, e_c)
\]

Defender 分支：

\[
M_B = G_B(z, M_A, e_{\mathrm{text}}, e_c)
\]

Defender 必须通过 cross-attention / causal interaction 模块显式接收 Attacker motion feature。

第一版优先复用 TIMotion 当前 interaction block，不要大改 Backbone。

重点做：

- structured combat condition embedding；
- attacker/defender role token；
- reaction-specific conditioning；
- optional role switching token。

---

## 6.4 第一版 Loss

沿用基础生成损失：

\[
L_{\mathrm{motion}}
\]

增加：

\[
L_{\mathrm{interaction}}
\]

其中第一版 \(L_{\mathrm{interaction}}\) 可直接来自：

- interaction feature consistency；
- action/reaction temporal alignment；
- role-conditioned classification loss。

建议先实现轻量版本，不要一次引入大量复杂损失。

---

## 6.5 评价指标

必须增加：

### Attack Accuracy
生成的攻击动作是否符合 attack_type。

### Defense Accuracy
生成的防御动作是否符合 defense_type。

### Reaction Success Rate

\[
RSR =
\frac{N_{\mathrm{correct\ reaction}}}
{N_{\mathrm{all}}}
\]

### Reaction Timing Error

若有 GT：

\[
E_{\mathrm{reaction-time}}
=
|t_{\mathrm{reaction}}^{gen}
-
t_{\mathrm{reaction}}^{GT}|
\]

---

# 7. 创新点 2：Combat Interaction Contact Guidance

## 7.1 要解决的问题

普通 interaction motion generation 即使视觉上看似：

- Punch；
- Block；
- Kick；
- Dodge；

三维空间里也可能没有真实满足格斗语义。

例如：

\[
d(Fist_A,Forearm_B) = 0.12m
\]

看起来像格挡，实际上没有发生有效接触。

---

## 7.2 方法依据

参考其他领域：

- ScoreHOI：contact-driven iterative refinement；
- Human-X：physically plausible interaction；
- MotionPRO：接触/压力信息对运动质量的重要性；
- Human-object interaction 中的 contact map / contact-aware refinement。

我们的创新不是“首次使用 Contact”。

真正贡献应定义为：

> Combat-specific attacker-defender spatio-temporal contact representation。

---

## 7.3 Combat Contact Map

定义：

\[
C_t^{ij}
\]

表示第 \(t\) 帧：

A 的身体部位 \(i\)

与

B 的身体部位 \(j\)

之间的格斗交互关系。

进一步增加 interaction type：

\[
Type \in
\{
Hit,
Block,
Avoid
\}
\]

因此：

\[
C_t^{combat}
=
(
BodyPart_A,
BodyPart_B,
Type,
Time
)
\]

---

## 7.4 三种主要约束

### Hit

例如：

\[
RightHand_A
\rightarrow
Torso_B
\]

要求：

\[
d(
RightHand_A,
Torso_B
)
\rightarrow 0
\]

在接触窗口发生。

---

### Block

例如：

\[
RightFist_A
\rightarrow
LeftForearm_B
\]

要求：

\[
d(
RightFist_A,
LeftForearm_B
)
\rightarrow 0
\]

同时攻击肢体不应直接穿过防御肢体。

---

### Avoid / Dodge

例如：

\[
RightFist_A
\not\rightarrow
Head_B
\]

要求：

\[
d(
RightFist_A,
Head_B
)
>
\delta_{\mathrm{safe}}
\]

---

# 8. Contact Loss

第一版定义：

\[
L_{\mathrm{contact}}
=
\lambda_{hit}L_{hit}
+
\lambda_{timing}L_{timing}
+
\lambda_{penetration}L_{penetration}
+
\lambda_{avoid}L_{avoid}
\]

## 8.1 Hit Loss

\[
L_{hit}
=
\sum_{t \in T_c}
\|
p_i^A(t)-p_j^B(t)
\|^2
\]

---

## 8.2 Timing Loss

\[
L_{timing}
=
|
t_{attack}
-
t_{contact}
|
\]

或：

\[
|
t_{block}
-
t_{attack}
|
\]

---

## 8.3 Avoid Loss

\[
L_{avoid}
=
\max
(
0,
\delta_{safe}-d_{ij}
)
\]

---

## 8.4 Penetration Loss

基于：

- body capsules；
- SMPL mesh；
- signed distance；
- joint spheres；

选一种项目中最容易实现的方法。

**MVP 优先使用 body-part sphere/capsule collision，不要第一版就做高成本 mesh SDF。**

---

# 9. Contact 数据如何获得

第一阶段不要求手工标注所有 Contact。

可以通过：

## 方法 A：已有 interaction dataset 自动推导

从 Inter-X / InterHuman 中：

1. 获取 body joint / mesh 位置；
2. 计算 pairwise body-part distance；
3. 根据阈值生成 pseudo-contact label；
4. 根据 textual description / interaction category 判断 Hit / Block / Avoid。

---

## 方法 B：GTA Combat

若数据可稳定获取：

- 作为 combat-specific contact 数据；
- 提取攻击和反应类别；
- 自动计算 contact pairs。

---

## 方法 C：视频恢复数据

后续扩展：

\[
Combat\ Video
\rightarrow
3D\ Motion
\rightarrow
Pseudo\ Contact
\]

但不作为第一阶段必须项。

---

# 10. 创新点 3：Dual-Robot Combat Physics Critic

## 10.1 要解决的问题

即使 Human Motion 满足：

- interaction；
- contact；

也不代表目标机器人能够执行。

原因包括：

- DOF 不同；
- joint limit；
- link length；
- mass distribution；
- velocity limit；
- actuator torque；
- balance；
- inter-robot collision。

因此：

\[
Human\text{-}plausible
\neq
Robot\text{-}executable
\]

---

# 11. 方法依据

主要参考：

## RobotMDM
使用 downstream controller performance / reward surrogate 评价生成 motion 的可执行性。

## CLAIMS
使用 tracking 与 physical metrics 形成生成—执行—反馈闭环。

我们的改造：

从：

\[
Single\ Motion\ Executability
\]

扩展为：

\[
\boxed{
Dual\text{-}Robot\ Combat\ Executability
}
\]

同时评价：

- tracking；
- balance；
- fall；
- combat contact；
- collision；
- penetration。

---

# 12. Physics Critic 输入输出

输入：

\[
E_\phi(
M_A,
M_B,
R_A,
R_B
)
\]

其中：

- \(M_A,M_B\)：双人格斗 motion；
- \(R_A,R_B\)：目标机器人参数。

输出：

\[
S_{\mathrm{physics}}
=
[
S_{\mathrm{track}},
S_{\mathrm{balance}},
S_{\mathrm{contact}},
P_{\mathrm{fall}},
P_{\mathrm{collision}}
]
\]

---

# 13. Physics Critic 训练数据生成

无需人工标签。

流程：

\[
Generated\ Human\ Motion
\]

↓

Retargeting

↓

\[
Robot\ Reference\ Motion
\]

↓

Isaac Lab

↓

Motion Tracking

↓

自动记录 Physics Labels。

---

# 14. 必须记录的 Physics Labels

## Tracking Error

\[
E_{\mathrm{joint}}
=
\frac{1}{T}
\sum_t
\|
q_t-q_t^{ref}
\|_2
\]

---

## Root Error

\[
E_{\mathrm{root}}
\]

---

## End-Effector Error

重点：

- fist；
- hand；
- foot；
- head；
- torso target。

\[
E_{\mathrm{ee}}
\]

---

## Fall Label

\[
y_{\mathrm{fall}}\in\{0,1\}
\]

---

## Joint Limit Violation

\[
V_{\mathrm{joint}}
\]

---

## Torque Violation

\[
V_{\tau}
\]

如果当前模拟器可获取 torque。

---

## Balance Score

可先使用：

- base tilt；
- COM；
- support stability；
- non-fall duration。

实现难度优先。

---

## Robot Contact Success

\[
CSR_{\mathrm{robot}}
\]

计算 Retarget + Tracking 后机器人是否仍在正确 body-part pair 上完成 contact。

这一项非常重要。

---

# 15. Physics Critic 模型

第一版优先小模型。

推荐二选一：

## Option A：Temporal Transformer
输入 motion sequence，输出 physics scores。

## Option B：Temporal MLP / 1D Conv
更简单，更容易训练。

不要第一版就使用大型 Transformer。

---

# 16. Physics Critic 第一阶段只用于 Candidate Reranking

不要 MVP 就把 critic gradient 塞进 diffusion 每一步。

第一版：

对于每个 prompt：

\[
K = 4 \sim 8
\]

生成候选：

\[
M_1,\ldots,M_K
\]

计算：

\[
Score_k
=
w_1S_{\mathrm{semantic}}
+
w_2S_{\mathrm{reaction}}
+
w_3S_{\mathrm{contact}}
+
w_4S_{\mathrm{physics}}
\]

选择：

\[
M^*
=
\arg\max_k Score_k
\]

只将 Top-1 或 Top-N 进入完整 Isaac Lab 验证。

这样显著降低计算成本。

---

# 17. 后续增强：Critic-Guided Generation

只有在 Candidate Reranking 证明有效后，再考虑：

\[
\nabla_M E_\phi(M)
\]

引入 sampling guidance。

形式：

\[
\epsilon'
=
\epsilon_\theta
-
\lambda_p
\nabla_M E_\phi(M)
\]

这属于第二阶段增强，不是 MVP 必须项。

---

# 18. 最终训练目标

第一版总体：

\[
L
=
L_{\mathrm{motion}}
+
\lambda_i L_{\mathrm{interaction}}
+
\lambda_c L_{\mathrm{contact}}
\]

Physics Critic 暂时用于：

\[
Reranking
\]

而不直接加入训练。

第二阶段再考虑：

\[
L
=
L_{\mathrm{motion}}
+
\lambda_iL_{\mathrm{interaction}}
+
\lambda_cL_{\mathrm{contact}}
+
\lambda_pL_{\mathrm{physics}}
\]

---

# 19. 输出数据格式

本课题最终核心输出不是 MP4。

而是：

\[
\boxed{
Two\text{-}Person\ Structured\ 3D\ Motion
}
\]

至少包含：

- root translation；
- root orientation；
- joint rotations；
- frame timestamps；
- actor / defender roles；
- combat condition；
- contact map；
- confidence / quality score。

建议额外保存：

```json
{
  "sample_id": "...",
  "text": "...",
  "attack_type": "...",
  "defense_type": "...",
  "target": "...",
  "role_a": "attacker",
  "role_b": "defender",
  "motion_a": "...",
  "motion_b": "...",
  "contact_map": "...",
  "semantic_score": 0.0,
  "reaction_score": 0.0,
  "contact_score": 0.0,
  "physics_score": 0.0,
  "selected": true
}
```

---

# 20. 视频只是可视化

需要提供 renderer：

\[
Motion
\rightarrow
SMPL/SMPL\text{-}X
\rightarrow
MP4
\]

目的：

- qualitative comparison；
- human evaluation；
- debugging。

不要把 video generation quality 当成主指标。

---

# 21. 五层评价体系

必须同时评价五个层次。

---

## Level 1：通用 Motion Generation Quality

用于和 TIMotion / InterGen 公平对比。

### FID ↓

生成动作与真实动作分布距离。

### R-Precision ↑

Text-Motion matching。

### MM Distance ↓

Text 与 Motion embedding 距离。

### Diversity

数据整体多样性。

### Multimodality

相同 Prompt 下生成结果多样性。

---

# 22. Level 2：格斗语义质量

## Attack Accuracy ↑

生成动作是否正确属于：

- jab；
- cross；
- hook；
- kick；
- etc.

---

## Defense Accuracy ↑

是否正确生成：

- block；
- dodge；
- step back；
- etc.

---

## Reaction Success Rate ↑

\[
RSR=
\frac{
N_{\mathrm{correct\ reaction}}
}{
N
}
\]

---

## Role Consistency ↑

攻击者、防守者、反击角色是否符合 prompt。

---

# 23. Level 3：Contact Quality

## Contact Success Rate ↑

\[
CSR=
\frac{
N_{\mathrm{valid\ contact}}
}{
N
}
\]

---

## Contact Distance ↓

\[
E_{\mathrm{contact}}
=
\min_t
\|
p_i^A(t)-p_j^B(t)
\|
\]

---

## Contact Timing Error ↓

\[
E_{\mathrm{time}}
=
|
t_c^{gen}
-
t_c^{gt}
|
\]

若没有 GT，可用 event detector / pseudo GT。

---

## Penetration Rate ↓

---

## Penetration Depth ↓

---

## Dodge Clearance ↑

\[
Clearance
=
d_{\min}
-
\delta_{safe}
\]

---

# 24. Level 4：Robot Executability

所有算法的生成 motion 必须经过同一套：

\[
Retargeting
\rightarrow
Motion\ Tracking
\rightarrow
IsaacLab
\]

公平比较。

评价：

### Tracking Success Rate ↑

### Joint Tracking Error ↓

### Root Tracking Error ↓

### End-Effector Error ↓

### Fall Rate ↓

### Joint Limit Violation ↓

### Torque Violation ↓

### Robot Contact Success ↑

### Inter-Robot Penetration ↓

---

# 25. Level 5：Synthetic Data Utility

这是整篇论文最终最重要的实验。

构建：

## Dataset A

\[
D_{\mathrm{real}}
\]

---

## Dataset B

\[
D_{\mathrm{real}}
+
D_{\mathrm{TIMotion}}
\]

---

## Dataset C

\[
D_{\mathrm{real}}
+
D_{\mathrm{Combat+Contact}}
\]

---

## Dataset D

\[
D_{\mathrm{real}}
+
D_{\mathrm{Ours}}
\]

其中 Ours：

\[
Combat
+
Contact
+
Physics
\]

用完全相同的 tracking / imitation policy 训练。

测试必须包含：

- seen combat motions；
- unseen combat motions；
- unseen combinations；
- perturbation；
- opponent distance variation。

最终比较：

- Tracking Success；
- Tracking Error；
- Fall Rate；
- Contact Success；
- Unseen Skill Generalization；
- Robustness。

必须证明：

\[
D_{\mathrm{real}}+D_{\mathrm{ours}}
>
D_{\mathrm{real}}+D_{\mathrm{generic\ synthetic}}
\]

才算真正证明生成示范数据有价值。

---

# 26. Human Evaluation

可以做 Blind Human Study 作为补充。

评价维度：

1. Motion Naturalness；
2. Combat Plausibility；
3. Attack-Defense Coordination；
4. Contact Realism。

使用 1–5 MOS 或 pairwise preference。

但不要把 human rating 作为核心结论。

---


# 26A. 动作捕捉/示范数据生成效果的完整评估协议

本课题生成的是可用于模仿学习的结构化 3D Motion / Reference Motion，而不是以“视频看起来好不好”为唯一目标。

因此评价必须回答五个层次的问题：

\[
\boxed{
动作像不像真实数据
}
\]

↓

\[
\boxed{
格斗语义对不对
}
\]

↓

\[
\boxed{
攻防接触是否正确
}
\]

↓

\[
\boxed{
目标机器人是否真正能执行
}
\]

↓

\[
\boxed{
这些数据是否真正提升模仿学习
}
\]

任何最终结果表都不能只报告 FID。

---

## 26A.1 第一层：分布质量与文本/动作一致性

这一层主要用于与 InterGen、TIMotion 等现有 human-human motion generation 文献公平比较。

必须严格复用官方 evaluation code、官方 feature extractor 和官方 test split。

至少报告：

| 指标 | 趋势 | 含义 |
|---|---:|---|
| FID | ↓ | 生成动作分布与真实动作分布之间的差异 |
| R-Precision Top-1/2/3 | ↑ | 生成动作与输入文本条件的语义匹配程度 |
| MM Distance | ↓ | Motion embedding 与对应 Text embedding 的平均距离 |
| Diversity | 接近真实数据 | 整个生成数据集的动作多样性 |
| Multimodality | ↑ | 同一个 Prompt 下不同生成样本的多样性 |

### FID

\[
FID =
\|
\mu_r-\mu_g
\|_2^2
+
Tr(
\Sigma_r+\Sigma_g
-
2(\Sigma_r\Sigma_g)^{1/2}
)
\]

其中：

- \(r\)：真实动作特征；
- \(g\)：生成动作特征。

注意：

**不得使用自己训练的 feature extractor 与论文官方数字直接比较。**

如果修改 feature extractor，则所有 baseline 必须重新计算。

---

## 26A.2 TIMotion 复现 Sanity Check

在开始 Combat 改造之前，必须先复现 TIMotion 原始任务。

以 TIMotion CVPR 2025 在 InterHuman 上论文报告结果作为 sanity reference。

论文报告的 TIMotion-RWKV 结果约为：

| Method | R@1 ↑ | R@2 ↑ | R@3 ↑ | FID ↓ | MM Dist ↓ | Diversity | MModality ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| InterGen | 0.371 | 0.515 | 0.624 | 5.918 | 5.108 | 7.387 | 2.141 |
| TIMotion-RWKV | 0.501 | 0.656 | 0.734 | 4.702 | 3.769 | 7.943 | 1.005 |

这些数值仅用于：

\[
\boxed{
Reproduction\ Sanity\ Check
}
\]

不能在以下情况直接声明“超过论文”：

- 数据预处理不同；
- 官方 split 不同；
- feature extractor 不同；
- evaluation code 不同；
- checkpoint 不同。

Codex 必须把：

- 官方复现值；
- 我们实际复现值；

分别保存。

建议表格：

| Method | Paper-reported | Our Reproduction | Relative Error |
|---|---|---|---|

如果关键指标偏差明显，优先修复 baseline，不能直接继续开发 Ours。

---

## 26A.3 第二层：格斗语义质量

标准 Human Motion Generation 指标无法回答：

> 生成的是不是真正要求的攻击、防御与反击动作？

因此新增 Combat-specific Semantic Metrics。

### Attack Classification Accuracy

定义独立动作分类器：

\[
f_{\mathrm{attack}}(M_A)
\rightarrow
\hat y_{\mathrm{attack}}
\]

动作类别至少包括：

- jab；
- cross；
- hook；
- uppercut；
- front kick；
- side kick。

计算：

\[
Acc_{\mathrm{attack}}
=
\frac{N_{\mathrm{correct}}}{N}
\]

---

### Defense Classification Accuracy

\[
f_{\mathrm{defense}}(M_B)
\rightarrow
\hat y_{\mathrm{defense}}
\]

类别例如：

- block；
- dodge-left；
- dodge-right；
- step-back；
- duck。

---

### Reaction Success Rate

对结构化条件：

\[
Attack_A
\rightarrow
Reaction_B
\]

判断 B 是否产生正确 reaction。

\[
RSR
=
\frac{
N_{\mathrm{correct\ attack-reaction\ pairs}}
}{
N_{\mathrm{samples}}
}
\]

---

### Role Consistency

对于：

\[
A: attacker,\quad B:defender
\]

检查动作是否发生 role confusion。

若 Prompt 包含反击：

\[
A\ attacks
\rightarrow
B\ blocks
\rightarrow
B\ counterattacks
\]

还应记录：

\[
RoleSwitchAccuracy
\]

---

### Reaction Timing Error

当数据存在 contact/reaction ground truth 或 pseudo-ground-truth：

\[
RTE =
|
t_{\mathrm{reaction}}^{gen}
-
t_{\mathrm{reaction}}^{gt}
|
\]

越低越好。

---

## 26A.4 第三层：格斗接触质量

这一层专门评价创新点 2。

必须区分：

\[
Hit,\quad Block,\quad Avoid
\]

不能把所有交互都统一成“距离越小越好”。

---

### Hit Contact Success Rate

对于目标：

\[
RightHand_A
\rightarrow
Head_B
\]

在指定接触窗口 \(T_c\) 内，若：

\[
\min_{t\in T_c}
d(
RightHand_A,
Head_B
)
<\delta_{\mathrm{hit}}
\]

则认为 Hit 成功。

\[
CSR_{\mathrm{hit}}
=
\frac{N_{\mathrm{hit\ success}}}{N_{\mathrm{hit}}}
\]

---

### Block Contact Success Rate

对于：

\[
Fist_A
\rightarrow
Forearm_B
\]

要求：

1. 正确 body-part pair 接触；
2. 接触发生在攻击到达目标之前或附近；
3. 攻击肢体不能直接严重穿透防御肢体。

\[
CSR_{\mathrm{block}}
=
\frac{N_{\mathrm{valid\ block}}}{N_{\mathrm{block}}}
\]

---

### Dodge / Avoid Success Rate

Dodge 不要求接触，而要求安全避让。

\[
d_{\min}
(
Fist_A,
Head_B
)
>
\delta_{\mathrm{safe}}
\]

定义：

\[
ASR_{\mathrm{dodge}}
=
\frac{N_{\mathrm{safe\ dodge}}}{N_{\mathrm{dodge}}}
\]

---

### Contact Distance

\[
E_{\mathrm{contact}}
=
\min_{t \in T_c}
\|
p_i^A(t)
-
p_j^B(t)
\|_2
\]

---

### Contact Timing Error

\[
E_{\mathrm{timing}}
=
|
t_c^{gen}-t_c^{gt}
|
\]

---

### Wrong-Body-Part Contact Rate

例如要求：

\[
Fist_A\rightarrow Forearm_B
\]

实际却：

\[
Fist_A\rightarrow Head_B
\]

定义：

\[
WBCR
=
\frac{
N_{\mathrm{wrong\ contact\ pair}}
}{
N
}
\]

这个指标对格挡场景非常重要。

---

### Penetration Rate / Penetration Depth

分别报告：

\[
PenetrationRate
\]

和：

\[
PenetrationDepth
\]

MVP 可以使用 body capsule / sphere 近似。

正式实验若 mesh collision 已稳定，再报告 mesh-level penetration。

---

## 26A.5 第四层：人体动作本身的物理合理性

为了与 PhysDiff 类文献建立可比性，补充以下物理指标：

### Penetrate ↓

人体最低 mesh vertex 进入地面的程度。

### Float ↓

应该接触地面但最低 body vertex 悬浮于地面的距离。

### Skate ↓

接触地面期间 foot joint 的水平滑动。

### Phys-Err ↓

参考 PhysDiff 思路：

\[
PhysErr
=
Penetrate
+
Float
+
Skate
\]

具体实现必须与所采用的 PhysDiff evaluation protocol 对齐。

注意：

这里评价的是：

\[
Human\ Motion\ Physical\ Plausibility
\]

还不能代替机器人可执行性评价。

---

## 26A.6 第五层：Robot Executability

所有生成方法必须经过完全相同的：

\[
Human\ Motion
\rightarrow
Retargeting
\rightarrow
Robot\ Reference
\rightarrow
Motion\ Tracker
\rightarrow
Isaac\ Lab
\]

才能公平比较。

严禁不同生成方法使用不同 tracker。

---

### Tracking Success Rate

一条动作若：

- 未跌倒；
- 完成主要动作；
- tracking error 不超过预设阈值；
- 格斗动作还需满足 contact condition；

则判定成功。

\[
TSR
=
\frac{N_{\mathrm{success}}}{N}
\]

---

### Mean Joint Tracking Error

\[
E_{\mathrm{joint}}
=
\frac{1}{TJ}
\sum_{t,j}
\|
q_{t,j}-q^{ref}_{t,j}
\|
\]

---

### Root Tracking Error

包括：

- root position；
- root orientation。

---

### End-Effector Error

重点报告：

- left/right hand；
- left/right foot；
- head；
- torso。

格斗场景特别报告：

\[
E_{\mathrm{fist}}
\]

和：

\[
E_{\mathrm{foot}}
\]

---

### Fall Rate

\[
FallRate
=
\frac{N_{\mathrm{fall}}}{N}
\]

---

### Joint Limit Violation Rate

\[
JLVR
=
\frac{
N(q<q_{\min}\ \lor\ q>q_{\max})
}{
N_{\mathrm{joint\ frames}}
}
\]

---

### Velocity Violation Rate

\[
VVR
=
\frac{
N(|\dot q|>\dot q_{\max})
}{
N_{\mathrm{joint\ frames}}
}
\]

---

### Torque Violation Rate

若 simulator 可可靠获取 actuator torque：

\[
TVR
=
\frac{
N(|\tau|>\tau_{\max})
}{
N_{\mathrm{joint\ frames}}
}
\]

---

### Robot Contact Success

Retarget 后必须重新计算。

Human Motion 中 Contact 正确，不代表 Robot Tracking 后仍然正确。

\[
CSR_{\mathrm{robot}}
\]

应作为核心指标之一。

---

### Inter-Robot Penetration

报告：

- collision count；
- penetration depth；
- invalid collision rate。

注意：

“预期格斗 Contact”不能被错误计为 invalid collision。

必须按照 Contact Map 区分：

\[
Expected\ Contact
\]

和：

\[
Unexpected\ Collision
\]

---

## 26A.7 第六层：Synthetic Demonstration Utility

这是最终最关键的评价。

不能因为生成动作 FID 更低，就得出“更适合模仿学习”。

必须通过下游训练验证。

固定：

- robot；
- tracker architecture；
- training budget；
- optimizer；
- simulator；
- seeds；
- test set。

只改变训练 demonstration dataset。

实验组：

### U0：Real Only

\[
D_{\mathrm{real}}
\]

### U1：Real + InterGen

\[
D_{\mathrm{real}}
+
D_{\mathrm{InterGen}}
\]

### U2：Real + TIMotion

\[
D_{\mathrm{real}}
+
D_{\mathrm{TIMotion}}
\]

### U3：Real + Combat Generator without Physics

\[
D_{\mathrm{real}}
+
D_{\mathrm{Combat}}
\]

### U4：Real + Ours

\[
D_{\mathrm{real}}
+
D_{\mathrm{Combat+Contact+Physics}}
\]

测试：

1. seen actions；
2. unseen action variants；
3. unseen attack-defense combinations；
4. unseen opponent distances；
5. push/perturbation；
6. morphology variation（若可行）。

核心指标：

- Tracking Success；
- Robot Contact Success；
- Fall Rate；
- Tracking Error；
- Unseen Combination Success；
- Robustness。

核心假设：

\[
U4
>
U2
>
U0
\]

但实验必须如实报告，不允许预设结果。

---

## 26A.8 数据规模公平性

Synthetic Data Utility 实验必须控制数据量。

例如：

\[
|D_{\mathrm{synthetic}}|
=
N
\]

所有生成方法都固定 N 条。

不能：

- TIMotion 加 1,000 条；
- Ours 加 10,000 条；

然后声称 Ours 更好。

建议至少做：

### Fixed-Count Comparison

所有方法相同 synthetic sample count。

### Fixed-Compute Comparison

所有方法相同生成/筛选预算。

### Scaling Curve

\[
N
\in
\{
0,
500,
1000,
2000,
5000
\}
\]

观察不同 synthetic data scale 下的 downstream performance。

这可以直接验证：

> 数据质量重要，还是数据数量重要？

---

## 26A.9 统计显著性

正式实验至少 3 seeds，推荐 5 seeds。

报告：

\[
Mean \pm Std
\]

对关键比较：

\[
Ours
\quad vs\quad
TIMotion
\]

和：

\[
Real+Ours
\quad vs\quad
Real+TIMotion
\]

建议增加：

- paired bootstrap；
或
- paired t-test / Wilcoxon signed-rank test，

具体根据 metric distribution 决定。

不能只展示单 seed 最优结果。

---

## 26A.10 定性评价

每种方法固定同一批 Prompt：

- jab → block；
- jab → dodge；
- hook → block；
- kick → step-back；
- attack → block → counterattack。

对每种方法：

- 相同随机种子；
- 相同角色初始距离；
- 相同动作长度；

输出：

1. SMPL animation；
2. contact visualization；
3. retargeted robot video；
4. failure case。

必须展示失败案例，而不是只挑最好看的结果。

---

# 26B. 与已有文献的对比实验设计

本课题不能把不同数据集、不同 motion representation、不同 simulator 下的论文数字直接放在一起声称 SOTA。

必须按照“可直接比较”和“机制级比较”分层设计。

---

## 26B.1 Comparison Track A：标准双人动作生成

### 目的

证明加入 Combat 模块后，没有破坏基础 human-human motion generation 能力，并与主流双人生成方法建立标准可比性。

### 数据集

优先：

- InterHuman official split；
- Inter-X official split。

### Baselines

必须优先复现：

1. **InterGen** — IJCV 2024；
2. **TIMotion** — CVPR 2025；
3. **ReGenNet** — CVPR 2024，若其任务接口可与 reaction setting 对齐；
4. Ours。

可选：

- ComMDM；
- RIG；

仅当官方代码/评估流程可以稳定复现时加入。

### 指标

严格沿用官方：

- R-Precision Top-1/2/3；
- FID；
- MM Distance；
- Diversity；
- Multimodality。

TIMotion 论文在 InterHuman 上对 InterGen、ComMDM、RIG 等进行了统一评价，因此这一赛道是最适合直接做主表的标准 benchmark。

### 结果表建议

| Method | R@1 ↑ | R@2 ↑ | R@3 ↑ | FID ↓ | MM Dist ↓ | Diversity | MModality ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real | | | | | | | |
| InterGen | | | | | | | |
| ReGenNet | | | | | | | |
| TIMotion | | | | | | | |
| Ours | | | | | | | |

注意：

Ours 不一定要求所有普通指标都绝对第一。

如果 Contact / Robot Executability 显著提升，但 FID 轻微变化，也可能是合理 trade-off。

---

## 26B.2 Comparison Track B：Action/Reaction 能力

### 目的

专门验证 Innovation 1。

### 文献 baseline

重点：

- ReGenNet；
- TIMotion；
- HERO（若输入模态差异过大，不做数值硬对比，可作为方法参考）；
- Human-X（若可在相同数据与 motion-only setting 下适配）。

### 数据

优先：

- Inter-X actor/reactor split；
- combat-filtered Inter-X；
- 后续自建 combat reaction test set。

### 统一输入

尽量统一成：

\[
Actor\ Motion
+
Condition
\rightarrow
Reactor\ Motion
\]

### 指标

- Reaction Success Rate；
- Reaction Timing Error；
- Defense Accuracy；
- Role Consistency；
- Standard motion metrics。

### 关键原则

HERO 输入 RGB video，而我们的主模型输入 motion/text，因此若不能构造完全相同输入，不允许用 HERO 的 published number 与 Ours 直接做 SOTA 数值比较。

可以：

- 引用其方法思想；
- 做 qualitative comparison；
- 或在统一 motion-input setting 下实现 adapted baseline。

---

## 26B.3 Comparison Track C：Combat Motion Benchmark

### 目的

验证模型是否真的生成格斗动作，而不是只在 generic interaction benchmark 上表现好。

### 数据集

优先：

- GTA Combat；
- Combat-filtered Inter-X；
- 若后续构建 Boxing/Sanda video-reconstructed dataset，则作为额外测试集。

### Baseline

重点：

1. **ActFormer** — ICCV 2023；
2. TIMotion；
3. InterGen；
4. Ours。

ActFormer 原论文在 GTA Combat 上使用 action-conditioned multi-person generation，并采用：

- Action Recognition Accuracy；
- FIDw；
- FIDm；

作为核心评价，因此在 GTA Combat 上应尽量复用同类 protocol。

### 指标

#### Action Recognition Accuracy ↑

使用固定预训练 action classifier。

#### Whole FID (FIDw) ↓

全部动作统一计算分布 FID。

#### Mean Class-wise FID (FIDm) ↓

对每一格斗类别单独计算 FID 后取平均：

\[
FID_m
=
\frac{1}{C}
\sum_{c=1}^{C}
FID_c
\]

对于格斗条件生成，FIDm 比单一整体 FID 更重要，因为可以防止模型只生成高频动作。

再增加我们的：

- Attack Accuracy；
- Defense Accuracy；
- Reaction Success；
- Contact metrics。

### 结果表建议

| Method | Action Acc ↑ | FIDw ↓ | FIDm ↓ | Reaction ↑ | Contact ↑ | Penetration ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ActFormer | | | | N/A/Adapted | N/A | |
| InterGen | | | | | | |
| TIMotion | | | | | | |
| Ours | | | | | | |

---

## 26B.4 Comparison Track D：Physics-aware Motion Generation

### 目的

验证 Innovation 2/3 是否真的改善物理质量。

### 文献依据

重点对照：

- PhysDiff — ICCV 2023；
- Morph — ICCV 2025；
- RobotMDM — SIGGRAPH Asia 2024；
- Human-X — ICCV 2025。

由于这些论文的：

- 数据集；
- 角色数量；
- motion representation；
- simulator；

不同，因此不能直接把 published number 与 Ours 放一张“谁更好”的 SOTA 表。

应该将它们的方法思想适配到同一 Combat Benchmark。

---

### Baseline P0：No Physics

TIMotion / Ours Generator 原始输出。

---

### Baseline P1：Post-hoc Physics Filtering

只做：

\[
Generate
\rightarrow
Physics\ Check
\rightarrow
Reject
\]

用于测试简单筛选是否已经足够。

---

### Baseline P2：PhysDiff-style Projection

如果工程可行，将 PhysDiff 的：

\[
Physics\ Projection
\]

思想适配到单人/双人 Motion。

至少复用其评价协议：

- Penetrate；
- Float；
- Skate；
- Phys-Err。

PhysDiff 论文明确采用这些物理指标，因此我们应至少保持这些指标，便于横向讨论。

---

### Baseline P3：RobotMDM-style Single-Motion Reward Surrogate

训练仅预测单个机器人 tracking quality 的 surrogate：

\[
E_{\mathrm{single}}(M)
\]

不使用：

- opponent interaction；
- combat contact。

这个 baseline 用于回答：

> 单机器人可执行性预测是否已经足够？

---

### Ours：Dual-Robot Combat Physics Critic

\[
E_{\mathrm{combat}}(M_A,M_B)
\]

联合预测：

- tracking；
- balance；
- fall；
- contact；
- collision。

### 结果表

| Method | Penetrate ↓ | Float ↓ | Skate ↓ | Tracking Success ↑ | Fall ↓ | Robot Contact ↑ |
|---|---:|---:|---:|---:|---:|---:|
| No Physics | | | | | | |
| Post-hoc Filter | | | | | | |
| PhysDiff-style | | | | | | |
| RobotMDM-style | | | | | | |
| Ours | | | | | | |

这张表是验证 Innovation 3 最重要的物理对比表。

---

## 26B.5 Comparison Track E：Closed-Loop / Data Selection

### 文献依据

CLAIMS CVPR 2026 已证明：

\[
Generation
\rightarrow
Tracking
\rightarrow
Physical\ Evaluation
\rightarrow
Data\ Iteration
\]

能够突破固定 motion dataset 的难度分布限制。

CLAIMS 论文在 PHC single-primitive tracker 上，仅使用约 AMASS 1/10 的数据量，在其测试设置中报告平均 failure rate 相比 baseline 降低约 45%。

该数字只能作为文献动机，不能与我们的 Combat 实验直接数值比较。

### 我们的公平实验

固定相同 candidate generation budget：

\[
K
\]

比较：

1. Random Selection；
2. Semantic Score Selection；
3. Contact Score Selection；
4. Simple Physics Filtering；
5. Ours Physics Critic Reranking。

所有方法最后保留相同数量：

\[
N
\]

条 demonstrations。

然后训练相同 tracking policy。

### 指标

- downstream tracking success；
- fall rate；
- contact success；
- unseen-skill generalization。

这一实验可以回答：

> 不是“生成更多数据”，而是“筛出更有训练价值的数据”是否更重要。

---

## 26B.6 Comparison Track F：最终 Downstream Imitation Learning

这是论文最重要的最终主实验。

### Training Sets

\[
D_0=D_{\mathrm{real}}
\]

\[
D_1=D_{\mathrm{real}}+D_{\mathrm{ActFormer/InterGen}}
\]

\[
D_2=D_{\mathrm{real}}+D_{\mathrm{TIMotion}}
\]

\[
D_3=D_{\mathrm{real}}+D_{\mathrm{Ours\ without\ Physics}}
\]

\[
D_4=D_{\mathrm{real}}+D_{\mathrm{Ours}}
\]

控制：

\[
|D_1^{syn}|
=
|D_2^{syn}|
=
|D_3^{syn}|
=
|D_4^{syn}|
\]

### Tracker

必须相同。

例如最终选择：

- BeyondMimic；
- engineai_rl_lab whole-body tracking；
- 其他固定 tracker。

不要在这个实验里同时改 tracking architecture。

### Test Sets

建议四组：

#### T1：Seen Motions
已见类别，新实例。

#### T2：Unseen Variants
同类动作但不同速度、方向、幅度。

#### T3：Unseen Combinations
例如训练见过：

- Jab→Block；
- Hook→Dodge；

测试：

- Jab→Dodge；
- Hook→Block→Counter。

#### T4：Physics Stress Test
加入：

- opponent distance change；
- friction variation；
- external push；
- initial pose perturbation。

### 主指标

| 指标 | 趋势 |
|---|---:|
| Tracking Success Rate | ↑ |
| Robot Contact Success | ↑ |
| Fall Rate | ↓ |
| End-Effector Error | ↓ |
| Unseen Combination Success | ↑ |
| Robustness under perturbation | ↑ |

---

## 26B.7 对比实验必须遵守的公平性规则

### 规则 1：相同数据 split

所有方法必须使用相同 train/val/test。

---

### 规则 2：相同 Synthetic 数量

不能用数据量优势替代方法优势。

---

### 规则 3：相同生成 Prompt

所有 text-conditioned baseline 使用相同 Prompt 集。

---

### 规则 4：相同初始两人距离与朝向

Combat Contact 对初始 geometry 很敏感。

---

### 规则 5：相同 Retargeting

所有 human motion 统一走同一个 retargeting pipeline。

---

### 规则 6：相同 Motion Tracker

保证 robot executability 的差异来自 demonstration，而不是 controller。

---

### 规则 7：相同 Simulation Settings

固定：

- timestep；
- friction；
- control frequency；
- PD gains；
- robot model；
- collision settings。

---

### 规则 8：Published Numbers 与 Reproduced Numbers 分开

表中明确区分：

- `Paper`
- `Reproduced`

不能将论文原值和本地重跑值混成同一种结果。

---

## 26B.8 建议论文最终主表

### Table 1：Standard Human-Human Motion Generation

对比：

- InterGen；
- ReGenNet；
- TIMotion；
- Ours。

指标：

- R-Precision；
- FID；
- MM Dist；
- Diversity；
- MModality。

---

### Table 2：Combat-specific Generation

对比：

- ActFormer；
- InterGen；
- TIMotion；
- Ours。

指标：

- Action Accuracy；
- FIDw；
- FIDm；
- Reaction Success；
- Contact Success。

---

### Table 3：Contact / Physical Plausibility

对比：

- TIMotion；
- Contact-only；
- PhysDiff-style；
- Ours。

指标：

- Contact Distance；
- Contact Timing；
- Penetration；
- Float；
- Skate。

---

### Table 4：Robot Executability

对比：

- Generic Synthetic；
- Contact-only；
- RobotMDM-style single critic；
- Ours dual-robot critic。

指标：

- Tracking Success；
- Joint Error；
- Fall；
- Robot Contact；
- Collision。

---

### Table 5：Synthetic Data Utility

对比：

- Real；
- Real + InterGen；
- Real + TIMotion；
- Real + Ours。

指标：

- Seen tracking；
- Unseen tracking；
- Unseen combination；
- Contact success；
- Fall；
- Robustness。

---

## 26B.9 哪些论文可以直接数值比较，哪些不能

### 可以直接数值比较

前提是完全复现官方 protocol：

- InterGen vs TIMotion on InterHuman；
- TIMotion / compatible methods on Inter-X；
- ActFormer-compatible methods on GTA Combat。

---

### 原则上不能直接用 published number 硬比较

由于任务不同：

- PhysDiff；
- RobotMDM；
- CLAIMS；
- Human-X；
- MotionPRO。

这些论文更适合：

1. 借用评价指标；
2. 实现 adapted baseline；
3. 在我们统一 Combat Benchmark 上重新跑。

不能写：

> “Ours 的 Fall Rate 低于 RobotMDM，所以优于 RobotMDM”

除非：

- 同一 robot；
- 同一 motion；
- 同一 tracker；
- 同一 simulator；
- 同一 evaluation protocol。

---

## 26B.10 评价协议实现验收标准

Codex 必须在进入正式训练前先实现完整 evaluator。

至少提供：

```text
evaluation/
├── standard_motion_metrics.py
├── combat_semantic_metrics.py
├── reaction_metrics.py
├── contact_metrics.py
├── human_physics_metrics.py
├── robot_tracking_metrics.py
├── downstream_utility_metrics.py
└── aggregate_results.py
```

若项目已有 evaluator，则增量扩展。

必须先使用：

- real test motions；
- 人工构造的 hit/block/dodge toy cases；

验证 evaluator 本身正确。

例如：

### Toy Hit
两只手明确相交，应：

\[
CSR_{hit}=1
\]

### Toy Miss
距离明显大于阈值，应：

\[
CSR_{hit}=0
\]

### Toy Dodge
最小距离超过安全 margin，应：

\[
ASR_{dodge}=1
\]

### Toy Penetration
人工构造严重 body overlap，应被 penetration metric 检出。

只有 evaluator 通过单元测试后，才能用于正式模型比较。

---


# 27. 数据集

优先级建议：

## 第一阶段

### InterHuman
用于双人生成 baseline。

### Inter-X
用于 actor/reactor、interaction、reaction。

---

## 第二阶段

### GTA Combat
用于 combat-specific 数据。

---

## 第三阶段

### Video-Reconstructed Boxing / Martial Arts
利用 WHAM / GVHMR 类方法扩充真实格斗 motion。

第一篇论文不要求一开始就完成视频数据扩充。

---

# 28. Retargeting 与机器人平台

第一阶段不要把主研究时间花在 Retargeting。

优先复用现有工具。

如果使用 Unitree G1：

- GMR；
- BeyondMimic / Isaac Lab tracking。

如果使用 EngineAI：

- 优先检查 engineai_rl_lab；
- PM01/T800；
- Isaac Lab whole-body tracking；
- MuJoCo Sim2Sim；
- Native SDK。

最终平台根据课题组真实硬件确定。

---

# 29. Isaac Lab 的作用

Isaac Lab 不负责生成 Human Motion。

它用于：

1. Robot Motion Tracking；
2. Physics Labels；
3. Dual-Robot Contact Evaluation；
4. Fall / Balance Evaluation；
5. Candidate Validation；
6. 后续 Sim2Real。

---

# 30. 机器人控制策略

Motion Tracking 策略在第一篇论文里应保持固定。

目的：

> 评价生成数据，而不是同时研究 controller。

因此：

- 所有不同数据集使用相同 tracker；
- 相同网络结构；
- 相同训练预算；
- 相同随机种子数量；
- 相同 simulator settings。

避免：

生成模型和 tracking policy 同时变化，导致无法判断收益来自哪里。

---

# 31. 消融实验

## A1：Ours - Combat Condition

移除结构化 combat condition。

验证：

- Attack Accuracy；
- Defense Accuracy；
- Reaction Success。

---

## A2：Ours - Contact Guidance

移除 Contact Module。

验证：

- Contact Success；
- Contact Distance；
- Penetration。

---

## A3：Ours - Physics Critic

不做 physics reranking。

验证：

- Tracking Success；
- Fall Rate；
- Joint Violation；
- Robot Contact Success。

---

## A4：Ours - Role Switching

不支持 attacker / defender role evolution。

验证：

- Counterattack Success；
- multi-stage combat interaction。

---

# 32. 核心实验矩阵

建议第一版先控制规模。

例如：

- 4 methods；
- 3 data settings；
- 3 seeds；
- 5–10 combat categories。

不要第一阶段做几十个任务。

建议优先动作：

1. right jab；
2. left jab；
3. hook；
4. front kick；
5. side kick；
6. block；
7. dodge；
8. attack→block；
9. attack→dodge；
10. attack→block→counterattack。

---

# 33. 第一阶段 MVP

MVP 不需要一次实现全部。

必须按以下顺序执行。

---

## Phase 0：项目审计

完成：

- repo structure；
- environment；
- existing model；
- dataset；
- config；
- logging；
- evaluation。

输出：

`implementation_audit.md`

---

## Phase 1：复现 TIMotion Baseline

目标：

- 正常训练 / 推理；
- 加载 pretrained；
- 输入 text；
- 输出 two-person motion；
- 渲染 MP4；
- 复现基本 FID / R-Precision。

验收：

- inference pipeline 稳定；
- 数据格式清晰；
- checkpoint 可保存/加载。

---

## Phase 2：Combat Dataset 子集

从现有数据中构建：

- attack；
- defense；
- reaction；
- contact；

相关子集。

生成结构化 CombatCondition。

验收：

- dataset loader 可返回 combat condition；
- 能统计每类样本数量；
- 无 silent label mismatch。

---

## Phase 3：Combat Condition Adapter

实现 Innovation 1 MVP。

验收：

- TIMotion 可以接收 structured condition；
- 同一 attack + 不同 defense 产生不同 reaction；
- Reaction Success 比原 TIMotion 有提升趋势。

---

## Phase 4：Contact Map + Contact Metrics

先不要修改模型。

先实现：

- automatic contact extraction；
- Hit / Block / Avoid representation；
- Contact Success；
- Contact Distance；
- Penetration；
- Dodge Clearance。

验收：

- 对真实/已有 motion 能正确计算 contact metrics；
- 可视化 contact pair。

---

## Phase 5：Contact Guidance

实现 Innovation 2。

优先：

- post-generation refinement；
- sampling guidance；
- lightweight contact loss。

不要第一版就重写 entire diffusion model。

验收：

- Contact Success 提高；
- FID 不显著恶化；
- Penetration 不明显增加。

---

## Phase 6：Robot Retargeting + Tracking

选择一个机器人平台跑通：

\[
Human\ Motion
\rightarrow
Robot\ Reference\ Motion
\rightarrow
Tracking
\]

验收：

- 至少 20–50 条生成动作可自动进入 simulator；
- tracking metrics 自动保存；
- fall/contact labels 自动保存。

---

## Phase 7：Physics Dataset

自动生成：

\[
Motion
\rightarrow
PhysicsLabels
\]

保存至少：

- tracking error；
- fall；
- joint violation；
- contact；
- balance。

---

## Phase 8：Physics Critic

训练轻量 critic。

验收：

- 能预测 fall；
- 能预测 tracking quality；
- physics score 与真实 simulator metric 有显著相关性。

---

## Phase 9：Candidate Reranking

每个 prompt：

\[
K=4\sim8
\]

生成候选。

使用：

\[
Semantic
+
Reaction
+
Contact
+
Physics
\]

选择最佳。

验收：

相较 random / first sample：

- Tracking Success ↑；
- Contact Success ↑；
- Fall Rate ↓。

---

## Phase 10：Synthetic Data Utility

训练相同 tracking policy：

- Real；
- Real + TIMotion；
- Real + Ours。

这是最终必须完成的实验。

---

# 34. 代码结构建议

仅当现有项目没有对应结构时使用。

```text
project/
├── configs/
│   ├── combat/
│   ├── contact/
│   └── physics/
├── data/
│   ├── combat_conditions/
│   ├── contact_labels/
│   └── physics_labels/
├── models/
│   ├── timotion/
│   ├── combat_condition/
│   ├── contact_guidance/
│   └── physics_critic/
├── retargeting/
├── simulator/
│   └── isaaclab/
├── evaluation/
│   ├── motion_metrics.py
│   ├── combat_metrics.py
│   ├── contact_metrics.py
│   ├── physics_metrics.py
│   └── downstream_metrics.py
├── scripts/
│   ├── train_baseline.py
│   ├── generate_combat_motion.py
│   ├── extract_contacts.py
│   ├── build_physics_dataset.py
│   ├── train_physics_critic.py
│   ├── rerank_candidates.py
│   └── evaluate_all.py
└── reports/
```

优先适配现有结构，不要机械照此新建。

---

# 35. 日志规范

每个实验至少记录：

```yaml
experiment_name:
git_commit:
seed:
dataset:
split:
backbone:
checkpoint:
combat_module:
contact_module:
physics_critic:
robot:
simulator:
date:
```

指标统一保存：

```json
{
  "fid": null,
  "r_precision": null,
  "mm_distance": null,
  "diversity": null,
  "attack_accuracy": null,
  "defense_accuracy": null,
  "reaction_success": null,
  "contact_success": null,
  "contact_distance": null,
  "penetration_rate": null,
  "tracking_success": null,
  "tracking_error": null,
  "fall_rate": null,
  "robot_contact_success": null
}
```

---

# 36. 随机种子

正式对比实验至少：

\[
3
\]

个 seeds。

最终论文推荐：

\[
5
\]

个 seeds。

第一阶段 debug 可以 1 seed。

---

# 37. 数据划分

必须避免：

- 同一原始 sequence 同时出现在 train/test；
- 同一视频切片泄漏；
- synthetic sample 派生自 test sample；
- contact label 生成过程使用 test annotation 训练 classifier。

优先：

- sequence-level split；
- subject-level split；
- source-video-level split。

---

# 38. 2×A6000 使用建议

GPU 0：

- TIMotion / generator training；
- contact module。

GPU 1：

- physics critic；
- Isaac Lab rollout；
- baseline parallel run。

不要一开始做模型并行。

优先：

- data parallel experiments；
- parallel seed scheduling；
- simulator parallel environments。

---

# 39. 不要第一版实现的内容

MVP 阶段不要做：

- 大型视频 foundation model 从零训练；
- end-to-end video-to-robot；
- contact force prediction；
- full differentiable simulator through diffusion；
- RL self-play；
- sim2real；
- 多机器人平台同时训练；
- 大规模 LLM / VLM reward model；
- 复杂 mesh-level contact solver。

这些作为第二阶段扩展。

---

# 40. 三个创新点的正式表述

## Innovation 1  
### Combat-conditioned Causal Reaction Generation

提出结构化格斗条件表示，将 attack type、attack target、defense type 和 attacker/defender role 显式融入双人动作生成，通过因果交互与动态角色条件建模 Attack–Defense–Counterattack 的反应关系。

解决：

> 通用双人动作生成虽然能产生互动，但缺乏格斗级因果反应逻辑。

---

## Innovation 2  
### Combat Interaction Contact Guidance

提出面向格斗语义的时空接触表示，将 Hit、Block 和 Avoid 三类关系编码为 attacker-defender body-part contact constraint，并通过 contact distance、timing、avoidance 和 penetration objective 引导生成。

解决：

> 动作视觉上像打中/格挡，但三维空间中没有真正满足有效接触。

---

## Innovation 3  
### Dual-Robot Combat Physics Critic

利用 Isaac Lab 自动生成 tracking、balance、fall、joint-limit、collision 和 combat-contact 标签，训练双机器人格斗可执行性 Critic，并利用该模型进行候选动作筛选和后续生成优化。

解决：

> 人体动作看起来合理，但目标机器人可能无法稳定跟踪或完成有效格斗接触。

---

# 41. 论文方法主线

最终论文应形成：

\[
\boxed{
Combat\ Condition
}
\]

↓

\[
\boxed{
TIMotion\ Backbone
}
\]

↓

\[
\boxed{
Causal\ Reaction
}
\]

↓

\[
\boxed{
Combat\ Contact
}
\]

↓

\[
\boxed{
Physics\ Critic
}
\]

↓

\[
\boxed{
Robot\text{-}Executable\ Demonstration
}
\]

而不是：

\[
TIMotion
+
ContactLoss
+
IsaacLab
\]

简单工程拼接。

---

# 42. 核心实验假设

## H1

结构化 Combat Condition 可以提高：

\[
Reaction\ Success
\]

而不显著降低基础 Motion Quality。

---

## H2

Combat Contact Guidance 可以提高：

\[
Contact\ Success
\]

并降低：

\[
Contact\ Distance
+
Penetration
\]

---

## H3

Physics Critic Reranking 可以提高：

\[
Tracking\ Success
\]

并降低：

\[
Fall\ Rate
\]

---

## H4

使用 Ours 生成的数据训练 Motion Tracking / Imitation Policy：

\[
D_{\mathrm{real}}+D_{\mathrm{ours}}
\]

能够显著优于：

\[
D_{\mathrm{real}}
\]

和：

\[
D_{\mathrm{real}}+D_{\mathrm{TIMotion}}
\]

---

# 43. 最终验收标准

Codex 不应以“代码能运行”作为完成标准。

至少达到：

## Baseline
- TIMotion inference/training 跑通；
- 标准 Motion Metrics 可计算。

## Innovation 1
- Combat Condition 可输入；
- Reaction Success 有提升趋势。

## Innovation 2
- Contact Map 可生成；
- Contact Metrics 可计算；
- Contact Success 有提升趋势。

## Innovation 3
- Isaac Lab 自动产生 Physics Labels；
- Physics Critic 可训练；
- critic score 与真实 physics outcome 有相关性；
- reranking 后 Tracking Success 提升。

## Final
- 完成 Real / Generic Synthetic / Ours 三组 downstream imitation-learning 对比；
- 至少 3 seeds；
- 输出完整表格；
- 输出消融；
- 输出失败案例；
- 输出定性视频。

---

# 44. 参考文献与方法依据

实现前请重新核实以下论文及其官方代码/发表状态。

## 基础动作生成与物理

1. Tevet et al.  
   **Human Motion Diffusion Model**  
   ICLR 2023.

2. Yuan et al.  
   **PhysDiff: Physics-Guided Human Motion Diffusion Model**  
   ICCV 2023.

3. Serifi et al.  
   **Robot Motion Diffusion Model: Motion Generation for Robotic Characters**  
   SIGGRAPH Asia 2024.

---

## 双人交互与反应

4.5. Xu et al.  
   **ReGenNet: Towards Human Action-Reaction Synthesis**  
   CVPR 2024.  
   重点用于 action→reaction 对比。


4. Liang et al.  
   **InterGen: Diffusion-Based Multi-human Motion Generation Under Complex Interactions**  
   IJCV 2024.

5. Xu et al.  
   **Inter-X: Towards Versatile Human-Human Interaction Analysis**  
   CVPR 2024.

6. Wang et al.  
   **TIMotion: Temporal and Interactive Framework for Efficient Human-Human Motion Generation**  
   CVPR 2025.

7. Yu et al.  
   **HERO: Human Reaction Generation from Videos**  
   ICCV 2025.

8. Ji et al.  
   **Towards Immersive Human-X Interaction: A Real-Time Framework for Physically Plausible Motion Synthesis**  
   ICCV 2025.

---

## Contact / Interaction

9.1. Diller and Dai.  
   **CG-HOI: Contact-Guided 3D Human-Object Interaction Generation**  
   CVPR 2024.  
   重点参考 joint contact modeling 与 inference-time contact guidance。


9. ScoreHOI  
   ICCV 2025.  
   重点参考 contact-driven refinement 思想。

10. MotionPRO  
    CVPR 2025.  
    重点参考物理接触/压力信息对于高质量 motion representation 的重要性。

---

## 自动数据闭环

10.5. Li et al.  
    **Morph: A Motion-free Physics Optimization Framework for Human Motion Generation**  
    ICCV 2025.  
    重点参考 synthetic noisy motion + physics refinement 的实验设计。


11. Xu et al.  
    **Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control (CLAIMS)**  
    CVPR 2026.

---

# 45. 最终 Codex 输出要求

每完成一个阶段，都必须更新：

1. `implementation_audit.md`
2. `experiment_log.md`
3. `results_summary.md`

若项目已有对应报告，则更新原文件，不要重复新建。

最终必须输出：

## A. Code Change Summary
- 修改文件；
- 新增文件；
- 核心接口；
- 配置变化。

## B. Reproduction Status
- 哪些 baseline 已跑通；
- 哪些 checkpoint 可复现；
- 哪些指标与原论文接近。

## C. Dataset Status
- 使用哪些数据；
- 数据规模；
- train/val/test；
- combat categories；
- contact label 来源。

## D. Experiment Results
- baseline；
- full model；
- ablation；
- multi-seed。

## E. Failure Analysis
必须分析失败动作，例如：
- contact miss；
- early block；
- late dodge；
- body penetration；
- retarget failure；
- robot fall；
- excessive torque。

## F. Final Recommendation
根据真实实验判断：
- 哪个创新有效；
- 哪个无效；
- 是否值得继续；
- 下一阶段优先做什么。

---

# 46. 最重要的一条

不要为了证明方案而强行让实验“涨点”。

如果某个模块没有改善：

- 明确报告；
- 分析原因；
- 保留日志；
- 不修改测试协议；
- 不删除失败结果。

最终目标是验证：

\[
\boxed{
Combat\ Causality
+
Combat\ Contact
+
Robot\ Physics
}
\]

是否真的能够产生比普通 synthetic motion 更适合 humanoid combat imitation learning 的示范数据。

