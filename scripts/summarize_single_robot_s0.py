"""Build an evidence-bound S0 report from completed immutable run artifacts."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=a.run
protocol=json.loads((r/'protocol.json').read_text())
audit=json.loads((r/'source_audit.json').read_text())
limits=json.loads((r/'limits_audit.json').read_text())
finished=json.loads((r/'launcher_finished.json').read_text())
assert finished=={'worker0_exit':0,'worker1_exit':0}
metrics={c['id']:json.loads((r/'results'/c['id']/'metrics.json').read_text()) for c in protocol['cases']}
assert len(metrics)==19
for name,m in metrics.items():
    assert m['training_steps_this_run']==0 and m['seed']==123 and m['actors']==1
    assert m['checkpoint_sha256']=='0ec24c54c1e1502f5c7bd27efb39ecc0630c06b5c01dc640a6d493d672fc515c'
    for ref in m['references']:
        if ref['sha256']:
            local=r/'references'/Path(ref['path']).name
            assert hashlib.sha256(local.read_bytes()).hexdigest()==ref['sha256']
    for path,digest in m['code_sha256'].items():assert protocol['code_sha256'][path]==digest
sources=protocol['sources']
arms=['full','prefix150','initial_ground_offset']
arm_names={'full':'完整原参考','prefix150':'前150帧','initial_ground_offset':'完整参考＋初始足地高度修正'}
summary={}
for arm in arms:
    group=[metrics[f'{s["id"]}_{arm}'] for s in sources]
    summary[arm]={'n':len(group),'no_fall':sum(x['no_fall_complete'][0] for x in group),
        'strict_pass':sum(x['strict_complete'][0] for x in group),
        'mean_strict_prefix_s':mean(x['strict_prefix_s'][0] for x in group),
        'total_seconds':sum(x['duration_s'] for x in group)}
gpu_rows=list(csv.reader((r/'gpu_samples.csv').open()))[1:]
gpu={}
for i in (0,1):
    values=[x for x in gpu_rows if x[1].strip()==str(i)]
    gpu[i]={'sampled_peak_memory_mib':max(int(x[2].split()[0]) for x in values),
            'sampled_peak_util_percent':max(int(x[4].split()[0]) for x in values)}
queues=[json.loads((r/f'queue_{i}.json').read_text()) for i in (0,1)]
start=min(x['started_at'] for q in queues for x in q)
end=max(x['started_at']+x['elapsed_s'] for q in queues for x in q)
gate=bool(metrics['standing']['strict_complete'][0]) and summary['full']['strict_pass']>=5
summary.update({'standing_pass':metrics['standing']['strict_complete'][0],'progression_gate_pass':gate,
    'gpu':gpu,'batch_wall_s':end-start,'completed_cases':len(metrics),
    'artifact_integrity_checked':True,'training_steps':0,'scope':'S0 only; no synthetic-data-utility claim'})
a.output.parent.mkdir(parents=True,exist_ok=True)
(a.output.parent/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 单机器人方案首轮服务器实验报告', '', '日期：2026-09-16｜阶段：S0 数据与固定执行基线验收', '',
'**结论：服务器实验已真实完成，但尚未通过进入 S1–S4 的验收条件。** 本次完成19项仿真（18项动作对照＋1项站立），另由独立执行者完成1项站立见证。完整原动作严格通过0/6；不能将前3秒的局部成功当作完整武术技能成功。本报告不是新生成算法的最终效果报告，训练步数为0。','',
'## 1. 已确定方案与本轮范围','',
'单机器人方案保留“阶段／部位条件扩充 → 技能保真约束修正 → 质量与覆盖选择 → 固定学习基线验证”。KungfuBot用于动作处理与执行验收的依据，MotionDreamer用于局部扩充的方法依据；本轮没有冒称复现这两篇论文的全部算法。先执行既定S0停止规则：若原始参考尚未通过固定基线，就不能将后续训练差异归为新数据的贡献。','',
'|阶段|本次状态|证据或条件|','|---|---|---|',
'|S0 源数据与执行验收|已运行，未通过|53文件完整性审计；6条完整动作、3组对照；19项仿真完成|',
'|S1 阶段条件数据扩充|未执行|尚无通过验收且完成技能阶段标注的种子集|',
'|S2 完整修正消融|未执行|本次仅做初始足地高度工程对照，非完整算法消融|',
'|S3 质量／覆盖选择|未执行|没有合格候选池，不能靠筛除难动作制造收益|',
'|S4 4组×3种子统一训练|未启动|S0未达标；不能宣称已验证数据训练价值|','',
'## 2. 数据、控制变量与复现条件','',
'数据为已有 KungFuAthleteBot 视频重建、GMR映射得到的G1参考。对冻结清单中的全部53文件检查SHA-256、有限数值、50Hz和36维结构。沿用原划分，不使用验证或测试集调参。首轮按清单顺序取前6个不同训练标题组的第一条，而非按成绩挑选。标题组不保证是不同演员或完全独立录制。','']
split=Counter(x['split'] for x in audit['motions'])
lines+=['源数据划分：'+ '，'.join(f'{k}={v}' for k,v in split.items())+'。本轮样本含长拳、南拳和太极套路；未把它们当作已标注的直拳、摆拳、前踢、侧踢测试集。','',
'固定验证器为官方TWIST预训练策略，提交 `42d8c134739eee51f28d7cc0ff72a86728afb8dc`；MuJoCo 3.3.7，Torch 2.5.1+cu121，环境规格 `86ed9bff`，服务器nlp03，两张RTX A6000。所有作业相同策略、随机种子123、50Hz控制／1kHz物理仿真。未改控制奖励、网络或策略权重。','',
'原数据为29个关节；本次按名称映射官方TWIST的25关节模型，明确省略左右腕pitch/yaw四关节。原始50Hz、时间长度和关节幅度保留，根四元数归一化、初始平面位置归零，重新计算FK和速度。此转换不等于KungfuBot原23受控自由度协议，亦不与历史IsaacLab结果直接作数值横比。','',
'三组：完整参考；前150帧（149次推进，即2.98秒）；完整参考＋常量高度偏移。最后一组只在初帧地面穿透超过1毫米时，将全段抬高“穿透深度＋2毫米”，不改关节角、速度、动作时长或相对运动。它只是工程诊断，不是阶段感知修正算法。','',
'## 3. 结果','',
'严格通过定义：**每一个执行帧**关节平均绝对误差<0.3rad、根位置误差<0.15m、选定身体点误差<0.15m，且骨盆高度≥0.35m。无跌倒只表示骨盆未低于0.35m，不能代替技能正确执行。阈值为本项目固定诊断阈值，并非原论文官方成功率。','',
'|参考组|仿真数|全程无跌倒|全程严格通过|平均首次失配前合格时长|','|---|---:|---:|---:|---:|']
for arm in arms:
    s=summary[arm];lines.append(f'|{arm_names[arm]}|{s["n"]}|{s["no_fall"]}/6|{s["strict_pass"]}/6|{s["mean_strict_prefix_s"]:.3f}秒|')
lines+=['','不同组时长不同，不能用“前150帧33.3% vs 完整0%”声称短动作方案更优秀；这揭示的是评价窗口差异。初始高度修正与完整组时长相同，但没有提高严格通过数，全程无跌倒反而从2/6变为1/6。','',
'|源ID|完整时长|原始完整：严格前缀／首次跌倒|高度修正：严格前缀／首次跌倒|前150帧严格通过|','|---|---:|---|---|---|']
for s in sources:
    full=metrics[s['id']+'_full'];repair=metrics[s['id']+'_initial_ground_offset'];prefix=metrics[s['id']+'_prefix150']
    def detail(m):
        fall='未跌倒' if m['first_fall_s'][0] is None else f'{m["first_fall_s"][0]:.2f}秒'
        return f'{m["strict_prefix_s"][0]:.2f}秒／{fall}'
    lines.append(f'|{s["id"]}|{s["full_seconds"]:.2f}秒|{detail(full)}|{detail(repair)}|{"是" if prefix["strict_complete"][0] else "否"}|')
lines+=['','235与292在前2.98秒严格通过，但完整执行分别在7.44秒、5.50秒首次跌倒。230抬高后由未跌倒变成3.86秒跌倒，说明“初始不穿地”不能替代动态可执行性验证。均值错误包含跌倒后帧，原始metrics同时保存首次跌倒和严格前缀，不利用跌倒前较小误差冒充成功。','',
'## 4. 源数据质量与失败解释','',
'|源ID|初始地面穿透|整体抬高量|超过关节限位0.01rad的帧占比|','|---|---:|---:|---:|']
for s,l in zip(sources,limits['motions']):
    assert s['id']==l['id']
    lines.append(f'|{s["id"]}|{s["initial_ground_penetration_m"]*1000:.1f}毫米|{s["constant_height_offset_m"]*1000:.1f}毫米|{l["any_violation_above_0p01_rad_frame_fraction"]*100:.2f}%|')
lines+=['','这些数据证明当前输入存在足地位置和目标模型关节范围问题，**未证明它们是失败的唯一原因**。还需区分参考质量、预训练策略覆盖、根速度／姿态命令、执行历史初始化及模型差异。站立通过只能排除部分基础部署错误，不能保证武术输入接口与分布完全匹配。','',
'先前只取150帧的做法不适合据此宣称完整套路可执行。今后以保留准备、发力、回收、收势的完整技能单元建库；不能机械截取固定秒数。关键阶段应独立复核，不把自动速度峰值当人工真值。','',
'## 5. 耗时、显存与统计边界','',
f'19项主队列总墙钟时间约{end-start:.1f}秒（两卡并行，不含数据准备、独立见证与下载）。显存每2秒采样：GPU0峰值{gpu[0]["sampled_peak_memory_mib"]}MiB，GPU1峰值{gpu[1]["sampled_peak_memory_mib"]}MiB；是整卡采样值，可能漏掉瞬时峰值。低显存来自预训练推理＋CPU物理仿真，**不能据此外推12次训练的显存或耗时**。','',
'只有6个训练来源组和单个确定性执行条件，不报告显著性、泛化收益或真实机器人成功率。重复站立不是独立训练种子。未使用成对接触真值、未验证目标击打力，也未生成新合成训练数据；C1新增有效覆盖、C2下游训练收益均未得到验证。','',
'## 6. 决策与下一步','',
'预先冻结的工程门槛为站立通过且至少5/6完整原动作严格通过，另需来源／技能标注验收。本次站立通过、原动作0/6，门槛失败，因此没有启动S4，不能将缺失训练结果写成方案成功。该5/6门槛是工程决策规则，不是统计功效结论。','',
'1. 先对完整拳／踢技能单元建立来源与阶段标签，并保留独立来源测试；当前6条混合套路只承担诊断。',
'2. 校准目标模型、参考速度处理及原始技能执行，逐项对照，不同时改控制器和数据。KungfuBot/PBHC逐参考复现可用于这一层，不能拿单动作策略证明统一数据效用。',
'3. 在可执行种子上实现阶段／支撑约束与速度幅度保护，再进行S1–S3；不得将单纯抬高、减速或减幅算作核心算法创新。',
'4. 通过验收后冻结验证器，按原计划执行Seed、Seed+Reuse、Seed+Generic、Seed+Ours四组×3训练种子，比较独立留出上的联合技能成功。','',
'## 7. 可追溯文件','',
'服务器目录：`/data/code/ouyanghao/oyh/humanoid-combat-motion/runs/single_robot_s0_20260916/`。',
'每例含`metrics.json`、`rollout.npz`和`scene.xml`；protocol记录源文件／代码／权重哈希，queue记录命令、状态、耗时，gpu_samples记录显存。汇总已逐例核查引用文件哈希、固定权重、代码一致性及19项完成状态。',
'本地保存完整归档`single_robot_s0_20260916.tar.gz`；参考数据、轨迹和日志未覆盖历史实验。新脚本：`single_robot_s0.py`、`inspect_single_robot_s0_limits.py`、`launch_single_robot_s0.sh`、`summarize_single_robot_s0.py`。',
'准备阶段曾因XML默认关节标签解析触发两次断言／字段异常，修正为既有源码的worldbody关节顺序后才生成冻结协议；两次均发生于创建参考和执行仿真之前，不属于隐藏的失败实验。','']
a.output.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
