"""Render a Chinese incremental experiment report from saved, auditable results."""
from pathlib import Path
import html
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'records/paired_v1/completed_snapshot'


def read(name):return json.loads((SNAP/name).read_text(encoding='utf-8'))


def main():
    adapter=[]
    for seed in range(3):
        record=read(f'timotion_adapter_fixedrng_seed{seed}_validation/evaluation.json')
        before=record['results']['zero_adapter']['mean_loss'];after=record['results']['trained_adapter']['mean_loss']
        adapter.append(f'<tr><td>{seed}</td><td>{before:.6f}</td><td>{after:.6f}</td><td>{(before-after)/before*100:.3f}%</td></tr>')
    tracking=[]
    for name,label in [('g1_tracking_eval_iter99_primed','100 次 PPO 更新'),('g1_tracking_eval_iter1098_primed','再增加 1,000 次更新'),('g1_tracking_eval_iter2000_primed','扩大并行环境后的中间模型 2000'),('g1_tracking_eval_iter3097_primed','本轮扩展训练最终模型 3097')]:
        if not (SNAP/name/'metrics.json').exists():continue
        r=read(name+'/metrics.json');times=[s*r['control_dt'] for s in r['first_failure_steps']]
        tracking.append(f'<tr><td>{label}</td><td>{r["environments"]}</td><td>{r["completion_fraction"]:.1%}</td><td>{statistics.mean(times):.3f} 秒</td><td>{min(times):.2f}–{max(times):.2f} 秒</td></tr>')
    dual=read('dual_g1_sample000_iter1098/metrics.json')
    quality=read('interhuman_validated_manifest/summary.json')
    official=(SNAP/'timotion_official_metrics_seed0_repeat1/evaluation.log').read_text()
    geometry=read('actor_reaction_test_seed123/geometry_audit.json')
    projected=read('actor_reaction_projected_contact_seed123/evaluation.json')
    reaction=read('actor_reaction_seed0/TRAINING_COMPLETE.json')
    projected_success=sum(bool(r['contact']['success']) for r in projected['cases'])/len(projected['cases'])
    latest=[]
    final_path=SNAP/'g1_tracking_eval_iter3097_primed/metrics.json'
    if final_path.exists():
        final=json.loads(final_path.read_text());errors=final['mean_alive_error']
        latest.append(f'<p>最终模型 3097：未触发终止比例 <b>{final["completion_fraction"]:.2%}</b>；身体相对姿态平均误差 {errors["aligned_body_error_m"]:.3f} m，世界坐标身体误差 {errors["body_error_m"]:.3f} m，根部误差 {errors["root_error_m"]:.3f} m。要求身体/根部平均误差均低于 0.15 m、关节平均误差低于 0.3 rad 且完成全段时，通过率为 {final["strict_tracking_pass_fraction"]:.0%}。这是本课题额外的空间筛查门槛，不是原论文官方成功定义。</p>')
        latest.append('<video controls preload="metadata" poster="media/learned_tracking_final_preview.png"><source src="media/learned_tracking_final.mp4" type="video/mp4"></video><p>左：参考；中：模型2000；右：最终模型3097。仍是同一训练参考和固定展示环境0，不能证明未见动作泛化。</p>')
    history_path=SNAP/'actor_reaction_history_test_seed123/evaluation.json'
    if history_path.exists():
        history=json.loads(history_path.read_text());hr=history['cases']
        latest.append('<h3>历史条件预测</h3><p>新增最多30帧已观察攻击者动作条件；共同地面只由历史确定，最后观察帧的前向速度/足接触位被清除，未来攻击者动作没有作为模型输入。接触类型和时间窗口仍由外部提供，这不是自主战术发现。</p>')
        for arm,label in [('inpainting_only','原模型历史固定'),('actor_conditioned','增加历史动作条件')]:
            error=statistics.mean(r['arms'][arm]['defender_position_rmse_m'] for r in hr)
            success=statistics.mean(float(r['arms'][arm]['contact']['success']) for r in hr)
            latest.append(f'<p>{label}：20个测试来源的未来防守位置 RMSE {error:.3f} m，几何接触通过率 {success:.0%}。单种子探索结果，不能宣称可靠战术因果或机器人收益。</p>')
    fk_path=SNAP/'actor_reaction_kinematic_interior_contact_seed123/evaluation.json'
    if fk_path.exists():
        fk=json.loads(fk_path.read_text())
        latest.append(f'<h3>保持骨段长度的可微接触优化</h3><p>进一步用骨段方向和根部位置作为优化变量，经正运动学恢复骨架，并把目标设为表面真正接触（0m），而非停在评价阈值边缘。20对的几何通过率为 <b>{fk["contact_success_fraction"]:.0%}</b>，最大骨长变化 {fk["maximum_bone_change_m"]:.2g} m。先前以3cm评价边界为优化目标的版本只有10%通过，两个版本均保留。这是同一测试批次启发的事后探索，<b>不是独立确认的最终算法成绩</b>，更不是机器人接触力成功率。</p>')
    audit_path=SNAP/'actor_reaction_kinematic_contact_g1ik/contact_audit.json'
    if audit_path.exists():
        ra=json.loads(audit_path.read_text())
        latest.append(f'<p>重定向后仅 {ra["mapped_cases"]}/20 个案例具有已核对的接触部位对应，近似几何通过率 {ra["proxy_success_fraction"]:.1%}；未映射的案例不计分。使用缩放的人体球体半径与机器人连接部位，检验接触是否保留，尚不能代替机器人碰撞网格与接触力。这与19/20人体通过率的分母不同，不能直接计算提升或下降幅度。</p>')
    multi=[]
    for label,pattern,total in [('训练参考','g1_tracking_eval_multiref_train20_*/metrics.json',20),('留出参考','g1_tracking_eval_multiref_test4_*/metrics.json',4),('接触优化双人参考','dual_g1_contact_*_multiref999/metrics.json',20)]:
        rr=[json.loads(p.read_text()) for p in sorted(SNAP.glob(pattern))]
        if rr:
            survival=statistics.mean(r['completion_fraction'] for r in rr)
            seconds=statistics.mean(statistics.mean(r['first_failure_steps'])*r['control_dt'] for r in rr)
            multi.append(f'<tr><td>{label}</td><td>{len(rr)}/{total}</td><td>{survival:.1%}</td><td>{seconds:.3f} 秒</td></tr>')
    if multi:
        latest.append('<h3>多参考控制器与泛化诊断</h3><p>相同原版PPO训练1,000次更新、512并行环境，共12,288,000环境步。20条训练片段来自7个视频标题组；4条测试片段来自另外1个标题组。未根据测试结果挑选数据。这里的“真实来源”是视频重建后重定向数据，不是光学MoCap。</p><table><tr><th>评价对象</th><th>已完成片段数</th><th>平均未触发终止比例</th><th>平均观测时长</th></tr>'+''.join(multi)+'</table><p>表中是一种控制器、一个训练种子的能力检查，不能当作三个数据集、三个种子的效用比较。固定起点与训练时随机片段起点也有难度差异，不能只用训练回报判断整段执行成功。</p>')
        latest.append('<video controls preload="metadata" poster="media/multireference_preview.png"><source src="media/multireference_diagnostic.mp4" type="video/mp4"></video><p>上排：第一个训练片段；下排：第一个留出片段。每排左侧为其自己的参考、右侧为实际策略轨迹。固定展示环境0，首次失配后冻结，不挑选成功案例。该视频同样只渲染记录状态，不是跨仿真器控制测试。</p>')
    attention_path=SNAP/'actor_reaction_crossattention_test_seed123/evaluation.json'
    if attention_path.exists():
        rr=json.loads(attention_path.read_text())['cases']
        latest.append('<h3>显式跨注意力反应模块</h3><p>防守者的逐帧查询读取攻击者历史帧；屏蔽观察窗口以外和查询时刻以后的攻击帧，未来输入扰动单元测试通过。它增加显式时序交互，仍不保证战术或物理正确。</p>')
        for arm,label in [('inpainting_only','历史固定基线'),('actor_conditioned','跨注意力条件')]:
            err=statistics.mean(r['arms'][arm]['defender_position_rmse_m'] for r in rr)
            contact=statistics.mean(float(r['arms'][arm]['contact']['success']) for r in rr)
            latest.append(f'<p>{label}：未来防守位置RMSE {err:.3f} m，近似接触通过率 {contact:.0%}。20个既有测试来源、一个训练种子的探索性结果。</p>')
    reaction_rows=''.join(f'<tr><td>{label}</td><td>{geometry["summary"][key]["proxy_contact_success_fraction"]:.0%}</td><td>{geometry["summary"][key]["defender_position_rmse_m"]:.3f} m</td></tr>' for key,label in [('inpainting_only','固定攻击者的原模型'),('actor_conditioned','增加攻击者动作条件'),('actor_conditioned_contact','再做软接触优化')])
    dual_rows=[]
    for group in ('generic','adapted'):
        records=[json.loads(p.read_text()) for p in sorted(SNAP.glob(f'dual_g1_{group}_sample_*_iter1098/metrics.json'))]
        if records:
            # All these saved diagnoses assert reference FPS == 50Hz control.
            duration=statistics.mean(statistics.mean(r['first_failure_steps'])*.02 for r in records)
            dual_rows.append(f'<tr><td>{group}</td><td>{len(records)}</td><td>{statistics.mean(r["completion_fraction"] for r in records):.0%}</td><td>{duration:.3f} 秒</td></tr>')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>人形机器人格斗项目：实际实验进度与结果</title><style>
body{font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#213047;background:#f4f7fb;margin:0}main{max-width:1120px;margin:auto;padding:36px 24px}h1{font-size:32px;line-height:1.4}h2{margin-top:38px;color:#12375b}nav,a{color:#1765a1}section,.card{background:white;border:1px solid #dce5ee;border-radius:12px;padding:24px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border-bottom:1px solid #dce5ee;padding:10px;text-align:left}th{background:#edf3f9}video,img{width:100%;border-radius:8px}pre{white-space:pre-wrap;font-size:13px;background:#eef3f8;padding:16px}.notice{border-left:5px solid #de9222;padding:14px;background:#fff6e6}.ok{color:#127246}.muted{color:#62748a}details{margin:14px 0}code{background:#edf3f8;padding:2px 5px} @media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}nav,video{display:none}details>*{display:block}}
</style><main><h1>人形机器人格斗项目：实际实验进度与结果</h1>
<p class="muted">2026-09-14 增量记录 · 原项目继续实施 · 所有数值来自服务器保存的实验输出</p>
<div class="notice"><b>目前尚未验证整套方案成功。</b>已完成预训练动作生成、一次官方指标评估、三个种子的关键词适配训练、离线攻击者条件实验、接触约束审计、G1 参考转换，以及 48 组双机器人物理跟踪诊断。“合成数据能提高模仿学习”还没有得到证据。</div>
<nav><a href="#data">数据</a> · <a href="#generation">生成与适配</a> · <a href="#tracking">机器人回放</a> · <a href="#dual">双机器人</a> · <a href="#remaining">剩余工作</a></nav>
<section id="data"><h2>1. 数据已经准备到哪一步？</h2>
<p>InterHuman 是同步双人动作数据。原始文件保持不变，按官方训练、验证、测试编号划分；训练不读取测试动作。结构检查要求至少 16 个原始帧，另外核对两人帧数一致、数值有限以及文件校验值。</p>
<table><tr><th>划分</th><th>通过结构检查的双人片段</th><th>关键词筛出的格斗候选</th></tr>__DATA_ROWS__</table>
<p>共排除 581 个过短片段。关键词候选只表示文字提及拳击、踢击、防御等，不能等同于准确的攻击者、防守者或接触标注。</p>
<p>动作压缩包来自官方地址；标注由传输镜像获取，50/50 个可取官方样本逐字节一致，其余未逐一核验。官方动作包内缺少编号 3945、4106 的动作，已经记录，未补造。</p></section>
<section id="generation"><h2>2. 动作生成与三次适配实验</h2>
<p><b>TIMotion</b> 是把文本变成双人动作的生成模型。本轮严格加载官方 Transformer 权重，并使用官方评估器及指标函数。一次评估加载 1,099 个可用测试片段，批量大小 96；官方不足 15 帧的排除逻辑与本项目训练的 16 帧门槛分别保留。</p>
<table><tr><th>指标</th><th>实测值</th><th>简单解释</th></tr><tr><td>FID</td><td>5.3755</td><td>生成与真实动作分布的差异；越低越好</td></tr><tr><td>R-Precision Top-1</td><td>0.4991</td><td>动作与描述匹配的检索表现；越高越好</td></tr><tr><td>Diversity</td><td>8.0917</td><td>生成动作差异程度，需结合质量看</td></tr><tr><td>Multimodality</td><td>0.9272</td><td>同一描述生成不同动作的程度</td></tr></table>
<p class="notice">这是一轮评估重复，论文默认 20 轮。日志中的零置信区间来自只有一次重复，<b>不代表结果没有不确定性</b>。不能作为完整论文表格复现成功的结论。</p>
<details><summary>展开官方函数产生的评估日志</summary><pre>__OFFICIAL__</pre></details>
<p><b>Condition Adapter（条件适配器）</b> 是接在模型上的小型可训练模块。三个随机种子各训练 1,000 步，固定原生成模型，仅学习格斗关键词条件。以下在 82 个验证候选上使用相同裁剪、扩散时刻和噪声比较。</p>
<table><tr><th>随机种子</th><th>未适配损失</th><th>适配后损失</th><th>相对下降</th></tr>__ADAPTER_ROWS__</table>
<p>改进幅度很小，并且使用了参与选模型的验证集；这不证明攻击—防御因果关系、生成质量或机器人学习性能提升。</p>
<p>另生成普通组与适配组各 24 个双人动作，使用相同提示词和随机噪声。人物编号连续性校正保留原始输出；它是几何后处理，不是物理优化。<a href="paired_generation_identity_corrected.html">查看之前的双人人体动画</a>。</p></section>
<section id="tracking"><h2>3. 真实训练策略的机器人回放</h2>
<p>下图和视频对比同一个训练源动作的目标姿态与 PPO 策略执行轨迹。<b>PPO</b> 是通过仿真反馈训练控制器的方法；<b>Tracking</b> 指让机器人跟随参考动作。</p>
<video controls preload="metadata" poster="media/learned_tracking_expanded_preview.png"><source src="media/learned_tracking_expanded.mp4" type="video/mp4"></video>
<p>左：参考动画；中：模型 1098；右：扩大到 1,024 个并行环境训练后的中间模型 2000。四倍慢放；首次失配后冻结画面，未把自动重置后的动作接进来。视频用 MuJoCo 绘制已记录的 Isaac 状态，<b>不是跨仿真器控制验证</b>。<a href="media/learned_tracking_diagnostic.mp4">较早失败版本也予以保留</a>。</p>
<table><tr><th>训练阶段</th><th>评估环境</th><th>未触发终止比例</th><th>平均观测时长</th><th>观测时长范围</th></tr>__TRACKING_ROWS__</table>
<p>目标时长 2.34 秒。模型 2000 有 19/32 个环境维持到参考末尾，但全局身体位置误差约 0.479 m、根部误差约 0.462 m。<b>存活不等于准确跟踪。</b>原版奖励同时包含身体相对姿态和全局锚点位置、朝向，但默认终止规则主要检查高度与朝向，不会严格排除水平漂移；格斗则需要更精确的双方空间关系。因此同时报告相对姿态误差和世界坐标误差，不能只看终止比例。这仍是同一训练参考诊断，不是未见动作泛化。</p></section>
<section id="reaction"><h2>4. 攻击者条件与接触约束：发现了什么？</h2>
<p>从源文件互不重叠的格斗候选中提取 139 个训练、10 个验证、20 个测试接触候选。这些是几何规则产生的 <b>Pseudo-label（伪标注）</b>，未经人工核验，也没有真实接触力；不能当作确定的攻击—防守语义。测试集中有 18 个格挡候选和 2 个击中候选。</p>
<p>新增的小型动作条件网络训练 1,000 步，验证损失从 __REACTION_BEFORE__ 降为 __REACTION_AFTER__。输入包含完整攻击者动作、文本和指定接触窗口，因此这是 <b>离线条件生成</b>，尚不是只看历史动作的在线因果反应。已知攻击者轨迹由程序固定，零偏差是输入约束，不是模型预测成绩。</p>
<table><tr><th>测试方法（20 对）</th><th>球体近似接触通过率</th><th>防守者位置 RMSE</th></tr>__REACTION_ROWS__</table>
<p class="notice">软优化把几何接触率提高到 60%，但个别骨段被拉长约 20.36 cm。加入“骨段变化不超过 2 cm”的筛查后，仅 15% 同时通过。这个结果暴露了优化器利用不真实身体变形完成目标，<b>不能写成物理成功</b>。</p>
<p>随后进行的探索性修正把骨段长度设为硬约束：最大长度变化约 0.00000012 m，接触通过率变为 __PROJECTED_SUCCESS__。这说明在保持身体结构时，现有条件生成和优化还难以完成目标。该修正由测试失败启发，属于事后诊断，不能当作预先确定的最终方法比较。骨长保持也不保证关节限位、平衡、速度或动力学。</p>
<p>Avoid（避让）尚缺可信的“若不躲避就会被击中”威胁标注，当前没有用简单距离变远来冒充避让成功。</p></section>
<section id="dual"><h2>4. 从双人人体动作到两台 G1</h2>
<p>已增加位置驱动的逆运动学基线：复用 GMR 的 G1 模型与 Mink 求解器，保持两人的共同世界坐标。它不假装复现 GMR 原版基于完整人体朝向的重定向。关节角没有超限，但部分关节瞬时速度仍过高，因此还不能直接作为合格机器人数据。</p>
<p>普通组与适配组各 24 对动作均已转换为 50 Hz 的参考文件；在每个场景内放置两台参与物理仿真的 G1，分别运行同一个已训练控制器，并记录手腕与对方躯干之间的接触力。</p>
<table><tr><th>组别</th><th>双人动作数</th><th>完成比例</th><th>平均首次失配时间</th></tr>__DUAL_BATCH_ROWS__</table>
<table><tr><th>首个双机器人测试</th><th>结果</th></tr><tr><td>场景数 / 每场景机器人</td><td>4 / 2</td></tr><tr><td>完成比例</td><td>__DUAL_SUCCESS__</td></tr><tr><td>首次失配步数</td><td>__DUAL_STEPS__</td></tr><tr><td>已记录手腕—躯干最大接触力</td><td>__DUAL_FORCE__ N</td></tr></table>
<p>该策略只训练过另一条单人参考动作，控制器能力不足与动作可执行性混在一起。当前结果只证明双机器人评估接口能够运行，暂不批准把这些失败作为物理评价网络的可靠训练标签。</p></section>
<section id="latest"><h2>6. 最新完成的增量结果</h2>__LATEST__</section>
<section id="remaining"><h2>7. 剩余工作与当前判断</h2>
<ol><li>完成多参考动作控制训练。已新增片段边界隔离接口，避免把不同动作错误拼成一条连续轨迹；单动作存活不能代替该项。</li><li>将离线攻击者条件网络发展为可验证的历史条件反应生成，并补充角色与反应标注。</li><li>解决 Hit（击中）、Block（格挡）、Avoid（避让）的时空约束，避免靠人体变形或失真距离完成接触。</li><li>采集有正负覆盖、可靠归因的双机器人执行记录，训练物理评价网络；控制器普遍失败的样本不能直接证明动作不可执行。</li><li>使用相同训练预算和至少三个种子，比较真实数据、真实＋普通生成、真实＋本方法；评估未见动作泛化。</li></ol>
<p><b>当前结论：</b>数据获取、动作生成、重定向、参考转换和双机器人物理评估的工程链路已连通；算法的最终实用性尚未验证。零完成率和小幅验证损失变化都按实记录，不包装成成功结果。</p>
<p><a href="https://github.com/Pendragon08/humanoid-combat-motion">GitHub 代码与实验记录</a> · 原始结果索引：<code>records/paired_v1/completed_snapshot/INDEX.json</code></p></section></main></html>'''
    rows=''.join(f'<tr><td>{name}</td><td>{quality["counts"][split]}</td><td>{quality["counts"][split+"_weak_combat"]}</td></tr>' for split,name in [('train','训练集'),('val','验证集'),('test','测试集')])
    for key,value in {'__DATA_ROWS__':rows,'__ADAPTER_ROWS__':''.join(adapter),'__TRACKING_ROWS__':''.join(tracking),
         '__REACTION_BEFORE__':f'{reaction["baseline_validation_loss"]:.6f}','__REACTION_AFTER__':f'{reaction["best_validation_loss"]:.6f}',
         '__REACTION_ROWS__':reaction_rows,'__PROJECTED_SUCCESS__':f'{projected_success:.0%}','__DUAL_BATCH_ROWS__':''.join(dual_rows),
         '__LATEST__':''.join(latest),
         '__OFFICIAL__':html.escape(official),'__DUAL_SUCCESS__':f'{dual["completion_fraction"]:.0%}',
         '__DUAL_STEPS__':str(dual['first_failure_steps']),'__DUAL_FORCE__':str(dual['maximum_recorded_wrist_torso_force_n'])}.items():doc=doc.replace(key,value)
    doc='\n'.join(line.rstrip() for line in doc.splitlines())+'\n'
    dest=ROOT/'reports/experiment_progress_20260914.html';dest.write_text(doc,encoding='utf-8',newline='\n');print(dest)


if __name__=='__main__':main()
