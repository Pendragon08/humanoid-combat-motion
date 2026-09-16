"""Incrementally update the existing HTML; never replace unrelated sections."""
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/engineai_single_v1'
def read(name):
    p=REC/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def main():
    refs=read('references_manifest.json');pd=read('pd_results.json');audit=read('transfer_audit.json')
    rows=[]
    if refs:
        for arm in ['raw_ik','joint_constrained']:
            samples=[r for r in refs['samples'] if r['arm']==arm]
            rows.append(f'<tr><td>{"普通重定向" if arm=="raw_ik" else "最终50Hz关节约束"}</td><td>{len(samples)}</td><td>{sum(r["speed_violation_fraction"] for r in samples)/len(samples):.3%}</td><td>{sum(r["acceleration_screen_violation_fraction"] for r in samples)/len(samples):.2%}</td><td>{max(r["peak_speed_rad_s"] for r in samples):.2f}</td></tr>')
    training=[]
    for name,label in [('warm_raw_ik','普通生成＋官方策略微调'),('warm_joint_constrained','约束生成＋官方策略微调')]:
        result=read(name+'_eval.json');complete=read(name+'_train_complete.json')
        errors=result['mean_alive_errors'] if result else {}
        training.append(f'<tr><td>{label}</td><td>{"100轮微调完成" if complete else "等待或进行中"}</td><td>{format(result["completion_fraction"],".1%") if result else "尚无结果"}</td><td>{format(result["strict_tracking_fraction"],".1%") if result else "尚无结果"}</td><td>{format(errors["joint_mae_rad"],".3f") if errors else "待评估"}</td><td>{format(errors["body_error_m"],".3f") if errors else "待评估"}</td></tr>')
    pdtxt=f'{sum(not r["fell"] for r in pd["samples"])}/{len(pd["samples"])} 条完成且未触发跌倒判据' if pd else '待最终50Hz数据执行结果'
    body='''<!-- ENGINEAI-SINGLE-BEGIN -->
<section id="engineai-single"><h2>众擎 PM01：双人生成方法向单人场景迁移</h2>
<p class="notice"><b>2026-09-16 工程试验。</b>复用双人 TIMotion 扩散结果，分别提取两个角色，映射为 PM01 单人空击参考。不是新训练了独立单人生成器，也尚未证明生成数据改善模仿学习。</p>
<p>3对×2角色=6条单人动作，两种约束处理共12条。单人场景取消人与人接触要求；本轮对照是关节约束，不是完整接触/动力学生成方法。全部历史开发样例保留，不按成功与否挑选。</p>
<p><b>平台：</b>官方 PM01 USD、24关节及电机配置；Isaac Sim4.5＋Isaac Lab2.1兼容接口。官方项目说明使用2.3.2，因此本轮不能称为完全相同环境复现。主验证与独立复核均完成200步真实物理推进和CUDA计算。</p>
<h3>参考数据质量</h3><table><tr><th>方案</th><th>动作数</th><th>速度超限率</th><th>加速度筛选超限率</th><th>最大关节速度 rad/s</th></tr>'''+''.join(rows)+'''</table>
<p>加速度上限100 rad/s²是研究筛选参数，不是厂家额定值。初次在30Hz修正后再插值到50Hz会重新超限，已保留初次结果，并把修正放到最终50Hz数据上。关节约束通过不意味着动力学执行通过。</p>
<h3>实际执行与学习</h3><p>简单PD物理诊断：'''+pdtxt+'''。失败只说明该简单控制器不能完成这些动作，不能推出动作对所有控制器不可执行。</p>
<table><tr><th>输入</th><th>PPO状态</th><th>固定起点完成率</th><th>严格跟踪通过率</th><th>关节误差 rad</th><th>身体位置误差 m</th></tr>'''+''.join(training)+'''</table>
<p class="notice"><b>本轮结论：平台与数据接口已跑通，生成数据执行验证未通过。</b>两组均完成等预算微调及32环境评价，但均0/32完成。关节约束满足不等于完整物理可执行；误差统计受首次终止时间影响，不能单独根据较小关节误差宣称方案改进。下一步首先排查根部轨迹、地面接触、初始化和参考适配；固定跟踪器，仅把它作为检验数据的工具。不能据此宣称所有控制器均无法执行，也不能宣称生成数据已经提高模仿学习。</p>
<p>官方发布策略完整示例校准32/32完成，严格全局跟踪2/32；从零训练300轮的官方示例对照0/32完成。因此停止未完成的冷启动生成组，保留日志，改为两组相同官方先验＋100轮微调。每组100×256×24=614,400环境步，仅一个训练种子。固定起点评价32个环境，首次失败后排除自动重置轨迹。它们是不同参考的单参考学习诊断，不是相同测试集上的数据效用对照。官方示例1220步，生成动作247步，完成率不能作为等难度横比。初次官方评估在500步被训练超时截断的结果保留但作废；完整评估把时限延长至参考之后。</p>
<h3>视频：参考与实际执行分开</h3><p>最终50Hz参考回放（只展示目标动作，不能作为成功证据）：</p><video controls preload="metadata" src="media/engineai_single/reference_50hz.mp4"></video>
<p>Isaac Lab实际PD执行轨迹（用SDK简化几何可视化；不是第二次MuJoCo仿真）：</p><video controls preload="metadata" src="media/engineai_single/isaac_pd_50hz.mp4"></video>
<p>实际PPO执行，固定环境0、第一次终止即停止，因很快失败所以视频较短。左侧普通重定向，右侧关节约束；未挑选最好环境：</p><video controls preload="metadata" src="media/engineai_single/isaac_ppo_raw.mp4"></video><video controls preload="metadata" src="media/engineai_single/isaac_ppo_constrained.mp4"></video>
<h3>建议采用的评估标准</h3>
<table><tr><th>层次</th><th>指标</th><th>用途</th></tr>
<tr><td>动作生成</td><td>动作语义正确率、FID、文本匹配、多样性</td><td>FID需匹配单人表示的评估器与足量独立数据，6条样例不报告FID</td></tr>
<tr><td>参考质量</td><td>关节/速度越界、脚滑、穿地、自碰撞、修正偏移</td><td>防止靠静止或过度平滑取得好分数</td></tr>
<tr><td>执行质量</td><td>完成率、首次失败时间、关节/根部/身体误差、跌倒、力矩饱和</td><td>必须真实物理步进，回放不算执行</td></tr>
<tr><td>训练价值</td><td>原始示范 vs 加普通生成 vs 加约束生成，在未见动作上比较</td><td>同算法同预算分别训练、多种子；这是论文最终证据</td></tr></table>
<p>严格阈值为本研究诊断设置：平均关节误差&lt;0.3rad、根部/身体位置误差&lt;0.15m且完成动作；不是统一行业标准。
学术依据：<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html">HumanML3D/CVPR2022</a>；<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PhysDiff_Physics-Guided_Human_Motion_Diffusion_Model_ICCV_2023_paper.html">PhysDiff/ICCV2023</a>。
平台：<a href="https://github.com/engineai-robotics/engineai_rl_lab">EngineAI官方仓库</a>。</p>
<p>细节、适用范围和后续正式实验见 <a href="../docs/engineai_single_experiment.md">实验说明</a>。原始记录位于 records/engineai_single_v1，代码与历史实验互不覆盖。</p></section>
<!-- ENGINEAI-SINGLE-END -->'''
    target=ROOT/'reports/experiment_progress_20260914.html';s=target.read_text(encoding='utf-8')
    s=re.sub(r'<!-- ENGINEAI-SINGLE-BEGIN -->.*?<!-- ENGINEAI-SINGLE-END -->\s*','',s,flags=re.S)
    s=s.replace('</nav>','</nav>\n'+body,1)
    if 'href="#engineai-single"' not in s:s=s.replace('<nav>','<nav><a href="#engineai-single">众擎PM01单人迁移实验</a> · ',1)
    target.write_bytes(s.encode('utf-8'));print('Updated existing report: #engineai-single')

if __name__=='__main__':main()
