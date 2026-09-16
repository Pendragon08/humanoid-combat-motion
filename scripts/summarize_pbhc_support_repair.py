"""Report fixed-policy execution against repaired AND untouched original targets."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_support_repair_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target=REC/'summary.json'
    if target.exists():raise FileExistsError(target)
    manifest=json.loads((REC/'manifest.json').read_text());rows=[]
    jobs=manifest['jobs'].copy()
    for job in json.loads((ROOT/'records/pbhc_support_diagnostic_20260917/manifest.json').read_text())['jobs']:
        raw=job.copy();raw.update(arm='raw',original_reference=raw['reference'],original_reference_sha256=raw['reference_sha256']);jobs.append(raw)
    tracked=['pelvis','torso_link','left_ankle_roll_link','right_ankle_roll_link','left_elbow_link','right_elbow_link','left_wrist_roll_link','right_wrist_roll_link']
    for job in jobs:
        folder=ROOT/job['output'];metrics=json.loads((folder/'metrics.json').read_text())
        assert sha(folder/'support_trace.npz')==metrics['trace_sha256']
        assert sha(ROOT/job['original_reference'])==job['original_reference_sha256']
        with np.load(folder/'support_trace.npz',allow_pickle=False) as z,np.load(ROOT/job['original_reference'],allow_pickle=False) as ref:
            valid=z['alive'].astype(bool);bn=list(z['body_names']);bids=[bn.index(n) for n in tracked]
            assert list(ref['body_names'])==bn and list(z['joint_names'])==list(ref['joint_names'])
            body=z['body_pos']-z['env_origins'][None,:,None,:]
            joint=np.abs(z['joint_pos']-ref['joint_pos'][1:,None]).mean(-1)
            root=np.linalg.norm(body[:,:,0]-ref['body_pos_w'][1:,None,0],axis=-1)
            berror=np.linalg.norm(body[:,:,bids]-ref['body_pos_w'][1:,None,bids],axis=-1).mean(-1)
            failure=z['termination_terms'].any(-1)
            passed=(joint<.3)&(root<.15)&(berror<.15)&(body[:,:,0,2]>=.35)&~failure&valid
            strict=float(passed.all(0).mean())
            # Raw arm is an independent recomputation of the original strict metric.
            if job['arm']=='raw':assert abs(strict-metrics['strict_every_frame_pass_fraction'])<1e-8
            row=dict(name=job['name'],arm=job['arm'],completion=metrics['completion_fraction'],
                repaired_target_strict=metrics['strict_every_frame_pass_fraction'],original_target_strict=strict,
                repaired_target_mean_root_m=metrics['mean_active_errors'][2],original_target_mean_root_m=float(root[valid].mean()),
                original_target_body8_m=float(berror[valid].mean()),original_target_joint29_rad=float(joint[valid].mean()),
                valid_env0_s=float(valid[:,0].sum()/float(z['fps'])),checkpoint_sha256=job['checkpoint_sha256'],
                reference_sha256=job['reference_sha256'],original_reference_sha256=job['original_reference_sha256'],trace_sha256=metrics['trace_sha256'])
            rows.append(row)
    rows.sort(key=lambda r:(r['name'],r['arm']))
    result=dict(scope='Four single-source development cases, frozen existing policies, one evaluation seed, nominal only. Not training gain, robustness, generalization or synthetic data utility.',
        original_target_note='Same untouched original geometry; execution termination is still the native environment evaluated with the supplied candidate reference. Report completion and censored means together.',
        rows=rows,script_sha256=sha(Path(__file__)))
    target.write_text(json.dumps(result,indent=2))
    lines=['# 准备阶段支撑修正：固定策略对照','',
        '4条开发动作，每条32个环境、seed123；固定原有2000更新检查点，无新增策略训练。仅无扰动条件。',
        'ground_align：全段常量高度对齐（约1.5cm）。prep_support：在此基础上只优化准备阶段12个腿部关节，单关节变化≤0.12rad，保持足部水平位置和姿态的软约束。技能阶段关节、时长、根XY、根朝向不变。',
        '严格阈值未变；同时报告针对修正目标和原始目标的结果，避免靠修改参考定义得到提升。均值仅含首次终止前及终止帧。','',
        '|动作|数据版本|完整执行|修正目标严格通过|原始目标严格通过|原始目标根误差(m)|',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["name"]}|{r["arm"]}|{r["completion"]:.2%}|{r["repaired_target_strict"]:.2%}|{r["original_target_strict"]:.2%}|{r["original_target_mean_root_m"]:.4f}|')
    lines+=['','本实验使用冻结的单动作策略，不能将失败解读为新参考不可学习；也不能将几何穿地减少等同于物理执行改善。原始随机扰动结果仍为主结果。本轮不据此直接启动大规模训练。']
    (REC/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
