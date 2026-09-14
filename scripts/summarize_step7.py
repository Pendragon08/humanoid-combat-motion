"""Summarize complete fixed-budget runs, retaining missing/error states."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics
import math
from datetime import datetime,timezone

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('records/step7/summary.json'))
p.add_argument('--html',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1]
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
rows=[];completed=[]
test_manifest=read(root/'runs/kungfu_test4_tracking_refs/manifest.json')
expected_refs={r['sample']:r['sha256'] for r in test_manifest['records']}
for arm in ('real','generic','contact'):
    for seed in (0,1,2):
        reuse=arm=='real' and seed==0
        train=root/('runs/multireference_kungfu20_seed0' if reuse else f'runs/step7_utility_{arm}_seed{seed}')
        row={'arm':arm,'seed':seed,'reused':reuse,'run':str(train.relative_to(root)),'status':'not_started','evaluations':[]}
        if (train/'config.json').exists():
            cfg=read(train/'config.json');row['status']='training_or_incomplete'
            if (cfg['seed'],cfg['environments'],cfg['iterations'],cfg['initial_checkpoint_sha256'])!=(seed,512,1000,None):raise ValueError('Unequal training budget/initialization')
            bundle=root/('data/tracking_seed/kungfu_train20_bundle.npz' if arm=='real' else f'runs/step7_bundles/{arm}.npz')
            if digest(bundle)!=cfg['bundle_sha256']:raise ValueError('Training bundle mismatch')
        if (train/'TRAINING_COMPLETE.json').exists() and (train/'model_999.pt').exists():
            row['status']='awaiting_evaluation';row['model_sha256']=digest(train/'model_999.pt')
            for index in range(4):
                sample=f'sample_{index:03d}'
                path=root/(f'runs/g1_tracking_eval_multiref_test4_{sample}_primed/metrics.json' if reuse else f'runs/step7_eval_{arm}_seed{seed}_{sample}/metrics.json')
                if not path.exists():continue
                m=read(path)
                if m['checkpoint_sha256']!=row['model_sha256'] or m['motion_sha256']!=expected_refs[sample]:raise ValueError('Evaluation provenance mismatch')
                if m['environments']!=32 or m['seed']!=123:raise ValueError('Unequal evaluation protocol')
                if not all(math.isfinite(float(v)) for v in m['mean_alive_error'].values()):raise ValueError('Nonfinite evaluation metrics')
                row['evaluations'].append({'sample':sample,'path':str(path.relative_to(root)),'sha256':digest(path),
                    'completion':m['completion_fraction'],'strict':m['strict_tracking_pass_fraction'],
                    'failure_time_s':statistics.mean(m['first_failure_steps'])*m['control_dt'],
                    'joint_error_rad':m['mean_alive_error']['joint_mae_rad'],'root_error_m':m['mean_alive_error']['root_error_m'],
                    'body_error_m':m['mean_alive_error']['body_error_m'],'aligned_body_error_m':m['mean_alive_error']['aligned_body_error_m'],
                    'return_to_first_failure':m['mean_alive_error']['reward']*m['valid_active_env_steps']/m['environments']})
            if len(row['evaluations'])==4:
                row['status']='complete'
                row['macro']={key:statistics.mean(r[key] for r in row['evaluations']) for key in ('completion','strict','failure_time_s','joint_error_rad','root_error_m','body_error_m','aligned_body_error_m','return_to_first_failure')}
                completed.append(row)
        rows.append(row)
groups={}
for arm in ('real','generic','contact'):
    members=[r for r in completed if r['arm']==arm]
    groups[arm]={'completed_seeds':[r['seed'] for r in members]}
    if len(members)==3:
        groups[arm]['metrics']={k:{'mean':statistics.mean(r['macro'][k] for r in members),'sample_std':statistics.stdev(r['macro'][k] for r in members)} for k in members[0]['macro']}
differences={}
if len(completed)==9:
    lookup={(r['arm'],r['seed']):r for r in completed}
    for treatment,control in [('generic','real'),('contact','real'),('contact','generic')]:
        diff=[lookup[treatment,s]['macro']['strict']-lookup[control,s]['macro']['strict'] for s in (0,1,2)]
        differences[treatment+' minus '+control]={'per_seed_strict_difference':diff,'mean':statistics.mean(diff),'sample_std':statistics.stdev(diff)}
critic_path=root/'runs/step7_coverage_critic/report.json'
if (root/'runs/step7_coverage_critic_v2/report.json').exists():critic_path=root/'runs/step7_coverage_critic_v2/report.json'
critic=read(critic_path) if critic_path.exists() else None
contact_audits={}
for arm in ('generic','contact'):
    path=root/f'runs/step7_{arm}_g1ik/contact_audit_toes.json'
    if path.exists():
        audit=read(path);contact_audits[arm]={'mapped_cases':audit['mapped_cases'],'robot_proxy_contact_fraction':audit['proxy_success_fraction'],'sha256':digest(path)}
result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'groups':groups,'completed_runs':len(completed),'expected_runs':9,'paired_differences':differences,'contact_retargeting_audit':contact_audits,
        'pilot_runs_complete':len(completed)==9,'full_step7_physics_pipeline_validated':False,
        'critic':{'report_sha256':digest(critic_path),'decision':critic['deployment_decision'],'results':[{k:r[k] for k in ('seed','validation_mae_s','test_mae_s','test_constant_baseline_mae_s','retrospective_reranking')} for r in critic['results']]} if critic else None,
        'interpretation':'Development pilot; C is contact-constrained but not physics-certified; single-robot utility, one held-out video group; positive or negative results do not validate paired robot physics.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if a.html:
    source=a.html.read_text(encoding='utf-8')
    start='<!-- STEP7-BEGIN -->';end='<!-- STEP7-END -->'
    table=''.join('<tr><td>'+html.escape(r['arm'])+'</td><td>'+str(r['seed'])+'</td><td>'+html.escape(r['status'])+'</td><td>'+('%.2f%%'%(100*r['macro']['strict']) if 'macro' in r else '待完成')+'</td><td>'+('%.2f%%'%(100*r['macro']['completion']) if 'macro' in r else '待完成')+'</td></tr>' for r in rows)
    section=start+'<section id="step7"><h2>第七步：筛选与合成数据效用验证</h2><p>已完成 '+str(len(completed))+'/9 次同预算控制器实验（含复用的真实来源种子0）。每次512环境、1000次更新、12,288,000环境步；统一4条开发留出动作。</p><table><tr><th>数据组</th><th>种子</th><th>状态</th><th>严格跟踪通过</th><th>全段未失配</th></tr>'+table+'</table><p>real：20条视频来源动作；generic：加20条普通生成防守动作；contact：加20条适配器＋骨长约束接触生成防守动作。新增条件来自训练集。contact尚未经过可靠的物理筛选，不等于物理可执行数据。四条留出片段属于同一视频标题组，且此前已用于开发诊断。</p>'
    if critic:
        section+='<h3>物理筛选门槛</h3><p>完整物理评价网络需要可靠的接触、跌倒、碰撞标签，当前尚不具备。本轮训练了三个种子的较小覆盖预测器，仅预测固定控制器何时首次失配；使用48条既有真实仿真记录，按提示词来源分组。</p><table><tr><th>种子</th><th>预测误差（秒）</th><th>常量基线误差（秒）</th></tr>'+''.join('<tr><td>'+str(r['seed'])+'</td><td>%.5f</td><td>%.5f</td></tr>'%(r['test_mae_s'],r['test_constant_baseline_mae_s']) for r in critic['results'])+'</table><p class="notice">不批准用于自动筛选可执行动作：原校准动作全都提前失败，K=6排序仍未产生完整执行成功的候选。预测失配时间不等于预测动作物理可行性。</p>'
    if differences:
        section+='<h3>每组3个控制器种子的平均结果</h3><table><tr><th>组别</th><th>严格通过率：均值±标准差</th><th>全段未失配：均值±标准差</th><th>首次失配/完成时间</th></tr>'+''.join('<tr><td>'+html.escape(k)+'</td><td>%.2f%% ± %.2f%%</td><td>%.2f%% ± %.2f%%</td><td>%.3f 秒</td></tr>'%(100*v['metrics']['strict']['mean'],100*v['metrics']['strict']['sample_std'],100*v['metrics']['completion']['mean'],100*v['metrics']['completion']['sample_std'],v['metrics']['failure_time_s']['mean']) for k,v in groups.items())+'</table>'
        section+='<h3>三个控制器种子的配对差值</h3><table><tr><th>比较</th><th>严格通过率平均差值</th><th>种子间标准差</th></tr>'+''.join('<tr><td>'+html.escape(k)+'</td><td>%.2f 个百分点</td><td>%.2f 个百分点</td></tr>'%(100*v['mean'],100*v['sample_std']) for k,v in differences.items())+'</table><p>这是固定预算下的小规模开发结果；不把3个种子或32个环境当作独立数据来源，也不由此宣称完整双人格斗方案成功。</p>'
    if contact_audits:
        section+='<h3>接触经过重定向后是否保留？</h3><p>从固定版本GMR模型核对并加入已有脚尖辅助坐标后，20条训练条件均可进行几何检查；旧版不含脚尖的审计结果仍保留。普通生成组重定向后近似接触通过率为 '+str(round(100*contact_audits['generic']['robot_proxy_contact_fraction'],1))+'%，接触约束组为 '+str(round(100*contact_audits['contact']['robot_proxy_contact_fraction'],1))+'%。这只是球体及辅助坐标的几何近似，不是碰撞网格或接触力认证。训练数据没有因此改变。</p>'
    if (a.html.parent/'media/step7_utility.mp4').exists():
        section+='<h3>相同留出动作的实际执行对比</h3><video controls preload="metadata" poster="media/step7_utility_preview.png"><source src="media/step7_utility.mp4" type="video/mp4"></video><p>左上参考、右上A真实来源、左下B普通生成、右下C接触约束。统一控制器种子0、第一条留出动作和环境0；首次失配后冻结，4倍慢放。MuJoCo只负责渲染Isaac实际轨迹，不是跨仿真器策略部署。</p>'
    section+='<p><a href="../records/step7/summary.json">完整结果与来源校验</a> · <a href="../docs/step7_protocol.md">实验协议及范围</a></p></section>'+end
    if start in source:source=source[:source.index(start)]+section+source[source.index(end)+len(end):]
    else:source=source.replace('<section id="data">',section+'<section id="data">',1)
    a.html.write_text(source,encoding='utf-8')
print('STEP7_SUMMARY',len(completed),'/9',flush=True)
