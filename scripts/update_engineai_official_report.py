"""Append official-version verification to the existing research report."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/engineai_official_v1'

def read(name):
    p=REC/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def main():
    storage=read('storage_after.json')
    rows=[]
    for name,label in [('official_full_eval','官方示例／官方策略'),('raw_ik_prior_eval','普通生成／未微调官方策略'),('raw_ik_eval','普通生成／100轮微调'),('joint_constrained_prior_eval','生成后关节约束／未微调官方策略'),('joint_constrained_eval','生成后关节约束／100轮微调')]:
        v=read(name+'_results.json')
        if v:
            rows.append(f'<tr><td>{label}</td><td>{v["completion_fraction"]:.1%}</td><td>{v["strict_tracking_fraction"]:.1%}</td><td>{v["mean_alive_errors"]["joint_mae_rad"]:.3f}</td><td>{v["mean_alive_errors"]["body_error_m"]:.3f}</td></tr>')
        else:rows.append(f'<tr><td>{label}</td><td colspan="4">尚无已完成结果，不能当作通过</td></tr>')
    status='已完成' if read('PILOT_COMPLETE.json') else '尚未全部完成；详见环境与运行记录'
    free=f'{storage["data"]["free_gib"]:.1f} GiB' if storage else '尚未采集'
    final_results=[read(k+'_results.json') for k in ['raw_ik_eval','joint_constrained_eval']]
    conclusion='结果尚未完整，暂不判断。'
    if all(v is not None for v in final_results):
        conclusion=('本轮两种生成动作在100轮微调后仍未完成整段跟踪。更换为官方指定Lab与原生RSL-RL未解决该样例失败；关节约束不能替代接触、重心运动及动态可执行性校验。但单样例短预算不能证明生成路线无效，也不能排除策略先验不适配或训练不足。下一步应先审查脚底接触、根部轨迹、动作重定向和初始状态，再决定扩展数据实验。' if all(v['completion_fraction']==0 for v in final_results) else '详见各组完成率；这仍是开发样例拟合诊断，需独立测试集和多种子才能判断数据训练价值。')
    videos=''.join(f'<figure><figcaption>{label}：Isaac实际物理状态记录，以SDK几何渲染；不是MuJoCo执行测试。生成组在首次失败处截断。</figcaption><video controls preload="metadata" width="640" src="media/engineai_official/{name}.mp4"></video></figure>' for name,label in [('official_full','官方示例'),('raw_ik','普通生成动作'),('joint_constrained','关节约束后动作')] if (ROOT/f'reports/media/engineai_official/{name}.mp4').exists())
    body='''<!-- ENGINEAI-OFFICIAL-BEGIN -->
<section id="engineai-official"><h2>众擎官方指定版本：环境校准与生成动作复测</h2>
<p>本节是上一轮兼容实验的后续。使用众擎明确指定的Isaac Lab提交c227752（项目称2.3.2），官方EngineAI代码、原生RSL-RL5.0.1和Torch2.7。复用该Lab版本支持的Isaac Sim4.5，官方没有锁定其Sim小版本，不能声称与作者完整机器环境相同。</p>
<p>本次安装前数据盘约142GiB可用，系统盘约15GiB；新环境、缓存及结果均在数据盘，旧环境和数据保留。最新采集数据盘可用：'''+free+'''。不重复安装整个Isaac Sim，不使用旧版网络结构转换或任务导入补丁。状态：'''+status+'''。</p>
<table><tr><th>验证条件</th><th>完成率</th><th>严格跟踪通过</th><th>关节MAE/rad</th><th>身体误差/m</th></tr>'''+''.join(rows)+'''</table><p><strong>结果解读：</strong>'''+conclusion+'''</p>'''+videos+'''
<p>完成率指完整参考时长内未触发官方物理终止；严格跟踪同时要求完成、平均关节误差&lt;0.3rad、根部及身体误差各&lt;0.15m。因此完成不等于精确跟踪。固定起点、32个评价环境；误差仅在首次终止前统计。生成组仍是第一条历史开发样例及其约束版本，每组微调100轮、256环境、614400步，仅一个训练种子。官方示例与生成样例长度、难度不同；不能直接横比为算法优劣。先验策略本身并非通用格斗跟踪器。严格指标是此前声明的研究诊断阈值，不是行业标准。</p>
<p>先完成环境物理步进、CUDA运算及独立复核，再评价完整官方示例。官方示例完成率低于90%时停止后续生成动作训练。仅关节约束通过不代表地面接触、平衡和动力学成立；本轮仍不构成“合成数据提升模仿学习”的独立测试证据。两组来自同一个双人生成样例的单人提取，约束组是生成后关节角度／速度／加速度修正，不是重新训练的物理感知生成模型。32个并行环境也不是32个训练随机种子。</p>
<p>核验文档：<a href="../docs/engineai_official_environment.md">官方指定版本环境</a>；原始日志与数据：records/engineai_official_v1。<a href="https://github.com/engineai-robotics/engineai_rl_lab">众擎官方框架</a> · <a href="https://github.com/isaac-sim/IsaacLab/blob/c22775241e28f465fe345fa1a482ad6d29d712b0/README.md">固定Lab提交兼容表</a>。</p></section>
<!-- ENGINEAI-OFFICIAL-END -->'''
    path=ROOT/'reports/experiment_progress_20260914.html';s=path.read_text(encoding='utf-8')
    s=re.sub(r'<!-- ENGINEAI-OFFICIAL-BEGIN -->.*?<!-- ENGINEAI-OFFICIAL-END -->\s*','',s,flags=re.S)
    s=s.replace('</nav>','</nav>\n'+body,1)
    if 'href="#engineai-official"' not in s:s=s.replace('<nav>','<nav><a href="#engineai-official">众擎官方版本复测</a> · ',1)
    path.write_bytes(s.encode('utf-8'));print('Updated #engineai-official')

if __name__=='__main__':main()
