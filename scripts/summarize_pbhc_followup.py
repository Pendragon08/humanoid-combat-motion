"""Aggregate all checkpoints and diagnostic regimes without best-case selection."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/pbhc_beyondmimic_20260916'
REC=ROOT/'records/pbhc_followup_20260916'
OUT=ROOT/'runs/pbhc_followup_20260916'
manifest=json.loads((REC/'manifest.json').read_text())
paths=list((BASE/'evaluation').glob('*/*/metrics.json'))
paths += [ROOT/job['output']/'metrics.json' for job in manifest['jobs']]
assert len(paths)==28 and all(p.exists() for p in paths)
rows=[];diagnostics=[]
for path in paths:
    m=json.loads(path.read_text());cp=Path(m['checkpoint']);name=cp.parent.name
    row=dict(name=name,regime=m.get('evaluation_regime','randomized'),
             updates=m['training_receipt']['completed_updates'],
             completion=m['completion_fraction'],strict=m['strict_every_frame_pass_fraction'],
             mean_pass=m['trajectory_mean_pass_fraction'],joint=m['mean_active_errors'][0],
             root=m['mean_active_errors'][2],body=m['mean_active_errors'][3],metrics=str(path),
             checkpoint_sha256=m['checkpoint_sha256'],reference_sha256=m['reference_sha256'])
    rows.append(row)
    with np.load(path.parent/'rollout.npz',allow_pickle=False) as z:
        mask=z['alive'].astype(bool);errors=z['errors'];limits=[.3,.3,.15,.15]
        breaches={}
        for j,label in enumerate(['joint29','joint23','root','body8']):
            first=[]
            for env in range(errors.shape[1]):
                ix=np.flatnonzero((errors[:,env,j]>=limits[j])&mask[:,env])
                first.append(float((ix[0]+1)/50) if len(ix) else None)
            breaches[label]=first
        delta=z['qpos'][:,:3]-z['reference'][:,:3];valid=mask[:,0]
        diagnostics.append(dict(name=name,regime=row['regime'],updates=row['updates'],
            first_threshold_breach_seconds=breaches,
            env0_root_axis_mae=np.abs(delta[valid]).mean(0).tolist(),
            caveat='Threshold breaches during alive interval; missing breach is not successful completion'))
rows.sort(key=lambda x:(x['name'],x['regime'],x['updates']))
(REC/'summary.json').write_text(json.dumps({'scope':'Development diagnostics only','rows':rows,'diagnostics':diagnostics},indent=2))
with (REC/'summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
names=sorted(set(r['name'] for r in rows));fig,axes=plt.subplots(1,2,figsize=(12,4))
for name in names:
    points=sorted([r for r in rows if r['name']==name and r['regime']=='randomized'],key=lambda x:x['updates'])
    axes[0].plot([p['updates'] for p in points],[p['completion'] for p in points],marker='o',label=name)
    axes[1].plot([p['updates'] for p in points],[p['root'] for p in points],marker='o',label=name)
axes[0].set_ylabel('Completion fraction (not strict success)');axes[1].set_ylabel('Root error before termination (m)')
for ax in axes:ax.set_xlabel('Policy updates');ax.grid(alpha=.3)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(REC/'learning_curve.png',dpi=150);plt.close(fig)
lines=['# 完整拳脚动作训练与后续诊断报告','',
       '4条动作各完成2,000次更新、49,152,000环境步，总196,608,000步。全部16组阶段评测与12组扰动诊断已收集。',
       '这是单参考开发校准，不是独立泛化测试，也不是合成数据效用证明。原始随机化结果保留为主结果；无扰动结果只解释失败来源。','',
       '|动作|评测条件|更新次数|完整执行率|逐帧严格通过率|根误差(m)|',
       '|---|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['name']}|{r['regime']}|{r['updates']}|{r['completion']:.2%}|{r['strict']:.2%}|{r['root']:.4f}|")
lines += ['','## 解释边界','',
          '- 完整执行只表示未触发原环境终止，不能代替严格跟踪。',
          '- 根误差均值仅统计首次终止前且包含终止帧；必须同时看完整执行率，避免将早失败造成的低误差当作更好。',
          '- no_push只去掉外力；nominal进一步去掉材质/质心/默认关节及初始扰动，保留参考初始速度。不是重新训练控制器。',
          '- nominal提高只能说明模型对扰动敏感，不能将其成绩冒充原随机化任务成绩。',
          '- 各动作仅一个训练种子；32环境不是32个独立动作来源。',
          '- 后续数据修正、生成器训练和数据效用对照需结合本报告判断条件，不因进程结束就宣布基线合格。','',
          '四条回放均固定环境0，展示实际记录状态，首次终止后冻结；不按成功结果选择样例。']
(REC/'report.md').write_text('\n'.join(lines)+'\n')
primary=[r for r in rows if r['regime']=='randomized' and r['updates']==2000]
next_status={'stage':'awaiting_evidence_review','all_primary_strict_zero':all(r['strict']==0 for r in primary),
             'data_utility_launched':False,'policy_training_added':False,
             'next':'Review disturbance ablation and failure times; fix demonstrated causes before data-utility experiments; keep all four sources and original metrics'}
(REC/'next_stage.json').write_text(json.dumps(next_status,indent=2))
index={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(REC/'metrics_index.json').write_text(json.dumps(index,indent=2))
print('FOLLOWUP_ANALYSIS_COMPLETE',len(rows))
