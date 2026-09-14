"""Generate a Chinese HTML results report from actual run files only."""
import csv
import html
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from combat.robot import ROOT

ARMS=['video_seed','generic_diffusion','joint_constrained','ground_refined','physics_selected']
LABELS={'video_seed':'视频恢复的原参考动作','generic_diffusion':'扩散模型生成变体','joint_constrained':'生成＋关节约束','ground_refined':'生成＋关节约束＋足部几何修正','physics_selected':'生成＋约束＋物理反馈筛选'}
METRICS=[('joint_limit_violation_fraction','关节角越界比例','%',100),('velocity_violation_fraction','关节速度超限比例','%',100),('acceleration_violation_fraction','加速度超筛选阈值比例','%',100),('acceleration_rms_rad_s2','关节加速度均方根','rad/s²',1),('joint_deviation_from_seed_rad','与种子动作的关节差异','rad',1),('fist_deviation_from_seed_m','拳部位置偏离种子','cm',100),('foot_sliding_proxy_m_s','推测接触期间足部滑动','m/s',1),('ground_penetration_peak_m','足部最大穿地深度的均值','cm',100),('self_collision_frame_fraction','发生自碰撞的帧比例','%',100),('pd_joint_rmse_rad','简单控制器跟踪误差','rad',1),('pd_fall','简单控制器下的摔倒比例','%',100),('pd_full_horizon_success','简单控制器下的完整跟踪成功','%',100),('strict_filter_pass','全部筛选条件通过比例','%',100)]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    out=ROOT/'deliverables';out.mkdir(exist_ok=True);(out/'media').mkdir(exist_ok=True)
    summaries=[read(ROOT/f'runs/pilot_seed{s}/summary.json') for s in [0,1]]
    training=[read(ROOT/f'runs/pilot_seed{s}/TRAINING_COMPLETE.json') for s in [0,1]]
    data=read(ROOT/'audit/data_audit.json')
    cleanup=[read(ROOT/'audit/system_cleanup.json'),read(ROOT/'audit/disk_cleanup.json')]
    archive=read(ROOT/'audit/tdmpc2_archive/cleanup_verified.json')
    allrows=[]
    for s in [0,1]:
        allrows.extend([{'training_seed':s,**r} for r in read(ROOT/f'runs/pilot_seed{s}/metrics.json')])
    with (out/'all_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    (out/'raw_results.json').write_text(json.dumps({'summaries':summaries,'training':training,'data':data},indent=2,ensure_ascii=False),encoding='utf-8')
    for p in (ROOT/'runs/pilot_seed0/media').iterdir():shutil.copy2(p,out/'media'/p.name)
    curves=[ [json.loads(x) for x in (ROOT/f'runs/pilot_seed{s}/loss.jsonl').read_text().splitlines()] for s in [0,1]]
    fig,ax=plt.subplots(figsize=(9,3.8))
    for s,rows in enumerate(curves):ax.plot([r['step'] for r in rows],[r['validation_x0_mse'] for r in rows],label=f'Training seed {s}')
    ax.set(xlabel='Training step',ylabel='Validation denoising MSE',title='Held-out-source validation loss');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'media/training_curve.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    keys=[('acceleration_violation_fraction','Acceleration threshold violations (%)',100),('joint_limit_violation_fraction','Joint-range violations (%)',100),('pd_fall','Falls under simple PD control (%)',100)]
    for ax,(key,title,scale) in zip(axes,keys):
        values=np.array([[r['arms'][arm][key] for arm in ARMS] for r in summaries])*scale
        ax.bar(np.arange(5),values.mean(axis=0),color=['#5078aa','#e79c41','#45a899','#776ac5','#c65460'])
        for row in values:ax.scatter(np.arange(5),row,s=12,color='#222222',zorder=3)
        ax.set_xticks(np.arange(5),['Seed','Diffusion','Joint','Ground','Physics'],rotation=20);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'media/constraint_comparison.png',dpi=160);plt.close(fig)
    table=''
    for key,name,unit,scale in METRICS:
        cells=[]
        for arm in ARMS:
            vals=[s['arms'][arm].get(key) for s in summaries];vals=[x for x in vals if x is not None]
            cells.append(f'{np.mean(vals)*scale:.3f}' if vals else '未计算')
        table+=f'<tr><th>{name}（{unit}）</th>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>'
    trainrows=''.join(f'<tr><td>{s}</td><td>{r["steps"]}</td><td>{r["seconds"]/60:.2f} 分钟</td><td>{r["peak_gpu_gib"]:.3f} GiB</td><td>{r["best_validation_x0_mse"]:.5f}</td></tr>' for s,r in enumerate(training))
    def metric(arm,key):return np.mean([r['arms'][arm][key] for r in summaries])
    fall=metric('ground_refined','pd_fall')*100
    seed_fall=metric('video_seed','pd_fall')*100
    acc0=metric('generic_diffusion','acceleration_violation_fraction')*100
    acc1=metric('ground_refined','acceleration_violation_fraction')*100
    outcome=f'加速度超过研究筛选阈值的比例从 {acc0:.2f}% 变为 {acc1:.2f}%；修正动作在简单控制器下仍有 {fall:.1f}% 的测试回合摔倒。原参考动作在同一控制器下也有 {seed_fall:.1f}% 的回合摔倒，因此需要先建立能稳定跟踪种子动作的学习控制器，才能可靠比较合成数据的作用。'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 格斗参考动作生成｜首轮实测</title><style>
    :root{{--ink:#172536;--muted:#566779;--blue:#215b83;--line:#dce4eb}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f9;color:var(--ink);font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1160px;margin:auto;padding:30px 26px 70px}}header{{padding:30px;background:#15334c;color:white;border-radius:16px}}h1{{font-size:29px;line-height:1.5;margin:5px 0}}h2{{font-size:23px;margin-top:36px;border-bottom:2px solid var(--line);padding-bottom:10px}}h3{{font-size:18px}}section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin-top:20px}}p{{margin:.65em 0}}a{{color:#16719f}}header a{{color:#bce3ff}}.badge{{display:inline-block;border-radius:20px;padding:3px 13px;background:#e0eff7;color:#174568;font-size:13px;margin:3px}}.callout{{border-left:5px solid #dc9441;background:#fff7e9;padding:16px 20px;margin:18px 0}}.good{{border-color:#3e8e7c;background:#eef8f4}}nav{{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:11px;text-align:left}}thead{{background:#eaf1f6}}.scroll{{overflow:auto}}video,img{{max-width:100%;border-radius:9px}}pre{{overflow:auto;background:#edf2f6;padding:16px}}small{{color:var(--muted)}}details{{border:1px solid var(--line);border-radius:8px;padding:14px;margin:14px 0}}summary{{cursor:pointer;font-weight:600}}@media print{{body{{background:white}}header{{background:white;color:#000}}section{{break-inside:avoid;border:0;padding:5px}}video{{display:none}}nav{{display:none}}}}
    </style><main><header><span class="badge">真实服务器实测</span><span class="badge">2 × RTX A6000</span><span class="badge">初步实验，尚非论文结论</span><h1>物理约束格斗参考动作生成<br>Unitree G1 首轮实验</h1><p>从公开视频恢复的武术动作出发，生成变体、修正约束，再分别检查动作回放与实际物理执行。</p></header>
    <nav><a href="#result">先看结果</a><a href="#video">动作视频</a><a href="#method">做了什么</a><a href="#metrics">全部指标</a><a href="#training">训练记录</a><a href="#cleanup">备份与清理</a><a href="#next">下一步</a></nav>
    <section id="result"><h2>本轮回答了什么</h2><p><strong>{outcome}</strong></p><div class="callout"><strong>不能把回放流畅等同于机器人已经学会。</strong><p>本轮训练的是动作生成网络，物理诊断使用简单的关节 PD 控制器（根据位置和速度误差施加电机力矩），没有训练带平衡能力的模仿学习策略。因此本轮不能证明合成数据提高了机器人模仿学习或实机性能。</p></div><p>关节范围等条件由求解器直接施加，其违规率下降只验证了实现，不构成方法创新或机器人执行成功的证据。所有数值来自服务器输出。失败案例保留，没有删除难例或只展示成功片段。</p></section>
    <section id="video"><h2>动作效果与物理诊断</h2><h3>① 参考动作回放：这段用于看生成的动作</h3><p>从左到右：视频恢复的原参考动作 → 扩散模型生成变体 → 加入关节约束与足部几何修正。三列播放时长一致。</p><video controls preload="metadata" poster="media/reference_preview.png" src="media/reference_comparison.mp4"></video><small>视频按给定姿态逐帧显示，属于 Reference Replay（参考动作回放）；它没有证明机器人靠电机稳定执行。</small><h3>② 实际物理仿真：这段用于看当前简单控制器能不能执行</h3><p>机器人受重力、地面接触和电机力矩限制，根部自由运动，没有外加“扶稳力”。画面中的 FALL 表示达到预先设定的摔倒判据；NO FALL FLAG YET 只表示尚未触发该判据，不代表稳定跟踪成功。</p><video controls preload="metadata" src="media/physical_pd_diagnostic.mp4"></video><small>这是对当前控制器与动作组合的诊断。失败不能直接推导为该动作在所有学习控制器下都不可执行。</small></section>
    <section id="method"><h2>已经实现的代码链路</h2><p>公开武术视频恢复的 G1 动作 → 统一关节顺序与帧率 → 小型扩散模型生成关节变体 → 固定时长的凸约束修正 → 足部接触几何修正 → MuJoCo 物理诊断 → 输出可继续接入跟踪框架的参考数据。</p><p>使用 {data['source_clip_count']} 条公开动作、{data['raw_frame_total']:,} 个原始帧；训练 / 验证 / 测试窗口分别为 {data['counts']['train']} / {data['counts']['validation']} / {data['counts']['test']}。窗口长度 64 帧，统一 30 帧/秒。每个窗口都先采用同一个初始高度校准规则：整体平移根部，使第一帧最低足部碰撞球接近地面；该常量平移同样作用于所有比较组，不是逐帧扶稳。验证与测试的源片段不进入生成器训练。</p><p>两个训练随机种子，每个在 {summaries[0]['cases']} 个生成样本上比较五种处理方法。测试来自四条长拳源片段，属于同一个留出套路，样本规模很小；不能据此声称跨演员或完整格斗类别泛化。</p><details><summary>“物理约束”在这一版中的具体范围</summary><p>关节角限制和电机力矩限制来自固定的 Unitree G1 模型；速度上限来自 GMR 的 G1 URDF（机器人结构描述文件）。加速度设为 100 rad/s²，是本轮研究筛选参数，并非已经核实的厂家上限。</p><p>QP（凸二次规划）尽量少改变动作，同时满足关节角、一阶差分速度和二阶差分加速度限制。足部修正只调整根部位置以改善种子动作的地面几何关系。未保证动态平衡、所有自碰撞、击打接触或接触力。第 5 组再生成三种噪声强度的候选，通过真实仿真与固定代价函数选择一个，是有限预算的物理反馈筛选；所有候选得分与失败保留。它使用更多仿真计算，而且筛选与评估使用同一简单控制器，不能解释为未见控制器上的泛化。</p><p>扩散模型保留种子根部路径，生成 29 个关节的动作变体；不是从文字直接生成全新机器人技能，也不是 RobotMDM 的正式复现。</p></details></section>
    <section id="metrics"><h2>逐组量化比较</h2><img src="media/constraint_comparison.png" alt="约束违规与物理摔倒的实测比较"><p>图中 Seed / Diffusion / Joint / Ground / Physics 对应下表五列；黑点是两个训练种子的结果。表格是两次运行的描述性平均，不是统计显著性检验。脚滑和足部接触由几何推测，没有真实接触标注。表中的足部穿地深度使用足底测量点估计；完整碰撞几何的穿透深度另见原始指标 collision_penetration_peak_m。</p><div class="scroll"><table><thead><tr><th>指标</th>{''.join('<th>'+LABELS[a]+'</th>' for a in ARMS)}</tr></thead><tbody>{table}</tbody></table></div><p>“全部筛选条件通过”同时要求位置、速度、加速度、自碰撞、穿地及当前控制器完整跟踪门槛通过。不能把未通过的数据包装成已经验证的可执行动作。</p><a href="all_metrics.csv">下载逐样本指标 CSV</a> · <a href="raw_results.json">完整原始结果与数据审计 JSON</a></section>
    <section id="training"><h2>训练记录</h2><img src="media/training_curve.png" alt="两个训练种子的真实验证损失"><p>Validation denoising MSE（验证集去噪均方误差）衡量网络能否从加噪的关节序列恢复动作，不是机器人跟踪奖励。检查点只按验证集选取。显存列记录 PyTorch 张量分配峰值，不包含 CUDA 上下文与缓存等进程开销。</p><table><thead><tr><th>随机种子</th><th>训练步</th><th>实际耗时</th><th>PyTorch 张量峰值显存</th><th>最佳验证误差</th></tr></thead><tbody>{trainrows}</tbody></table><p>软件：PyTorch 2.5.1+cu121、MuJoCo 3.3.7、OSQP 1.0.5。环境通过真实 CUDA 矩阵计算、物理步进与渲染检查，并由独立检查过程按文档复核。</p></section>
    <section id="cleanup"><h2>运行前的磁盘清理与 TD-MPC2 归档</h2><p>先清理已确认可重建的软件安装包与索引缓存：系统盘约 {cleanup[0]['removed_bytes']/1e9:.2f}GB，数据盘约 {cleanup[1]['removed_bytes']/1e9:.2f}GB。随后按照你的要求，把旧 TD-MPC2 项目完整归档到现有私有 GitHub 仓库，校验通过后才删除服务器原目录，数据盘可用空间增加约 {archive["reclaimed_bytes"]/1e9:.2f}GB。</p><p><a href="https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/server-archive-20260914">TD-MPC2 完整归档与恢复附件</a>。本轮删除证据见 <a href="cleanup_verified.json">清理校验记录</a>。</p><p>当前格斗项目代码、数据、模型和结果均保存在独立目录，保留其他用户的文件及无关研究项目。</p></section>
    <section id="next"><h2>接下来怎样发展成你的课题</h2><div class="callout good"><strong>最重要的下一步：训练统一的 G1 动作跟踪策略。</strong><p>在相同训练预算下，比较“种子动作”“种子＋普通生成动作”“种子＋约束生成动作”，观察测试动作上的跟踪误差、摔倒率和泛化变化。这一步才能回答 Synthetic Motion（合成动作）是否真正改善 Imitation Learning（模仿学习）。</p></div><p>再把学习控制器的失败反馈传回生成器，加入攻击—防御角色与接触时刻/接触部位条件，逐步覆盖双人格斗。当前代码保留了这条扩展路线的数据入口，但没有把尚未实现的模块算作结果。</p><p>这轮的意义是让数据、生成、约束、仿真和可视化链路落地，并用真实失败定位接下来需要解决的问题。</p></section>
    <section><h2>来源和实现参考</h2><ul><li><a href="https://github.com/NPCLEI/KungFuAthleteBot">KungFuAthleteBot：视频恢复的武术参考动作</a>；<a href="https://huggingface.co/datasets/LuluCao/KungfuAthleteBot">公开数据</a></li><li><a href="https://github.com/unitreerobotics/unitree_mujoco">Unitree 官方 MuJoCo 机器人模型</a></li><li><a href="https://github.com/YanjieZe/GMR">GMR：人体到机器人动作重定向</a></li><li><a href="https://github.com/GuyTevet/motion-diffusion-model">MDM：动作扩散的相关方法，当前小型网络并非其复现</a></li><li><a href="https://osqp.org/docs/interfaces/python.html">OSQP 凸约束求解器</a>；<a href="https://mujoco.readthedocs.io/en/stable/python.html">MuJoCo 物理仿真接口</a></li></ul><p><a href="humanoid_combat_motion_pilot_code_and_results.zip">下载项目代码、模型、数据与结果 ZIP</a> · <a href="code_bundle_verification.json">下载包校验信息</a>。完整启动命令和限制说明见包内 README。此报告由实际实验文件自动生成。</p></section></main></html>'''
    (out/'humanoid_combat_motion_pilot_report.html').write_text(body,encoding='utf-8')
    shutil.copy2(ROOT/'audit/tdmpc2_archive/cleanup_verified.json',out/'cleanup_verified.json')
    print('REPORT_COMPLETE',out/'humanoid_combat_motion_pilot_report.html')

if __name__=='__main__':main()
