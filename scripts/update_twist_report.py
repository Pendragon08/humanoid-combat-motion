"""Incrementally replace the tracker-plan block with evidence and actual outcomes."""
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    s=json.loads((ROOT/'records/twist/summary.json').read_text(encoding='utf-8'))
    groups=s['groups'];rows=s['results']
    table=''
    labels={'kungfu_train20':'此前选定的4条训练参考','kungfu_test4':'4条开发集参考（同一来源组）','dual_generic':'24组通用生成双人动作','dual_adapted':'24组适配生成双人动作'}
    for key,g in groups.items():
        table+=f"<tr><td>{labels[key]}</td><td>{g['cases']}</td><td>{g['strict_complete_cases']}/{g['cases']}</td><td>{g['no_fall_complete_cases']}/{g['cases']}</td><td>{g['cases_with_initial_penetration_over_1cm']}/{g['cases']}</td></tr>"
    details=''
    for r in rows:
        details+=f"<tr><td>{html.escape(r['case'])}</td><td>{r['strict_complete']}</td><td>{r['no_fall_complete']}</td><td>{', '.join(f'{x:.3f}' for x in r['joint_mae_rad'])}</td><td>{', '.join(f'{x:.3f}' for x in r['root_error_m'])}</td><td>{r['initial_max_penetration_m']*100:.1f}</td></tr>"
    section=f'''<!-- TRACKER-REPLACEMENT-AUDIT-BEGIN -->
<section id="tracker-replacement"><h2>已执行：TWIST官方预训练跟踪器与双机器人测试</h2>
<p><b>2026-09-15：已部署并跑完57项固定测试。</b>包括1项双机器人站立对照、8条原有单人参考、48组原有双人生成参考。另完成单机器人站立和独立复核。代码已接入现有项目，原BeyondMimic结果保留；本轮没有重新训练策略，也没有改写参考动作。</p>
<p>使用<a href="https://github.com/YanjieZe/TWIST/tree/42d8c134739eee51f28d7cc0ff72a86728afb8dc">TWIST官方固定版本</a>及通用预训练权重。<b>Pretrained controller（预训练控制器）</b>是已经学过动作跟踪的网络；每20毫秒根据目标动作与自身状态输出关节控制，MuJoCo每1毫秒执行一次真实物理步。不是把动画直接播放到机器人上。</p>
<table><tr><th>测试组</th><th>样本数</th><th>严格整段成功</th><th>全程未跌倒</th><th>初始穿插&gt;1厘米</th></tr>{table}</table>
<p>双人组的“成功”和“未跌倒”要求<b>两台都满足</b>。严格成功要求整段每帧：保留的25个关节平均误差&lt;0.3弧度、根部位置误差&lt;0.15米、8个身体部位平均世界坐标误差&lt;0.15米，同时骨盆高度不低于0.35米。未跌倒只是较弱的稳定性指标，不能替代正确跟踪。穿插按官方碰撞几何统计，包含穿地、自身和对方；不是已证实的真实机器人穿透量。</p>
<p class="notice"><b>结论：新控制器已能执行部分原有武术动作，但双人严格跟踪仍未解决。</b>不能把站立通过、动作像武术或双方发生碰撞写成格斗成功。现有生成→位置重定向链路尚未提供普遍适合直接执行的双人参考；模型碰撞几何差异也需要审计。这轮没有证明合成数据提高模仿学习。</p>
<h3>双机器人同场景视频</h3>
<p>以下画面均来自同一个物理场景中的两台受控机器人，蓝色与灰色用于区分。视频由实际仿真状态重新渲染；不是分屏拼接，也不是目标姿态回放。</p>
<p><b>工程对照：两台机器人站立5秒。</b>用于验证两个策略实例、历史状态和共同物理环境；没有攻击/防御动作。</p><video controls preload="metadata" src="media/twist_dual_standing.mp4"></video>
<p><b>双人适配生成样本000：失败实例。</b>展示初始几何问题和后续失稳，不能当作成功演示。</p><video controls preload="metadata" src="media/twist_dual_adapted_000.mp4"></video>
<p><b>双人适配生成样本002：不跌倒不等于严格跟踪。</b>保留相对运动偏差供检查；此视频展示的是控制器诊断，不代表攻防语义或接触准确。</p><video controls preload="metadata" src="media/twist_dual_adapted_002.mp4"></video>
<p><b>单人南拳样本010。</b>作为控制器能力对照，不能替代上面的双人任务。</p><video controls preload="metadata" src="media/twist_single_010.mp4"></video>
<h3>接口和可靠性核验</h3>
<ul><li>官方策略23个动作输出，两个腕部旋转关节由目标直接交给PD（按位置误差施加关节力矩），合计25个执行关节。旧参考的4个腕部俯仰/偏航自由度在官方模型中固定，按名称显式记录遗漏，不假装完整29自由度跟踪。</li>
<li>官方源码的观测、10步历史、动作缩放、PD参数和力矩限幅被复用。与实际官方run()连续200个物理步核对，最终观测差异和qpos（机器人位置与关节角状态）差异均为0；已修正float32状态转换细节。</li>
<li>两台机器人各自拥有历史状态，输入共同世界坐标不独立居中；启用官方碰撞几何。碰撞力只说明发生物理接触，不代表正确击中/格挡。</li>
<li>本轮输入沿用已有NPZ的速度值；没有经过官方MotionLib额外19帧平滑。官方数据样例位于外部下载，本次以官方默认站立命令做基础验证；没有声称已跑完其原论文评测集。</li></ul>
<h3>为什么还不能直接宣称优于BeyondMimic？</h3>
<p>此前每项BeyondMimic只训练12.288M环境步，约为上游配置训练上限的1/240；上限并非必须训练量。本轮同时改变了预训练来源、仿真器、机器人关节数、起始随机扰动和评价身体集合，因此只能说<b>现有项目获得了一个可运行的更广覆盖候选控制器</b>，不能把百分比差异归因于算法本身。预训练数据与我们的数据是否重合尚未核实。</p>
<h3>下一步聚焦什么？</h3>
<ol><li>保留TWIST作为可执行性诊断后端；先在同一MuJoCo几何中排除初始穿地、自身穿插，并增加支撑脚约束。每次修正保存前后误差和动作语义，避免只靠降低难度提高成绩。</li><li>对两个身体联合重定向，约束相对位置、可达性、攻防时序和接触；先让短时低速格挡/闪避动作通过，再处理高速碰撞。</li><li>只有可靠双人参考和跟踪基线成立后，再固定训练预算比较真实数据与真实＋生成数据，评价未知动作泛化。</li></ol>
<details><summary>展开全部57项结果（不隐藏失败）</summary><table><tr><th>case</th><th>各机器人严格成功</th><th>各机器人未跌倒</th><th>关节MAE/弧度</th><th>根误差/米</th><th>初始穿插/厘米</th></tr>{details}</table><p>这里的平均误差覆盖完整时段，包含跌倒之后；原始JSON另外保存首次跌倒时间、跌倒前关节误差和连续严格成功时长。不要将全程均值当作删去失败后的成绩。</p></details>
<p>复现实验记录：<a href="../records/twist/summary.json">完整指标与来源校验值</a>；<a href="../records/twist/parity.json">官方代码一致性</a>；<a href="../records/twist/independent_witness_v2.txt">独立执行见证</a>。环境复用，不安装完整Isaac Gym训练栈；官方源码/模型约59 MB。</p>
</section><!-- TRACKER-REPLACEMENT-AUDIT-END -->'''
    target=ROOT/'reports/experiment_progress_20260914.html';text=target.read_text(encoding='utf-8')
    text,n=re.subn(r'<!-- TRACKER-REPLACEMENT-AUDIT-BEGIN -->.*?<!-- TRACKER-REPLACEMENT-AUDIT-END -->',lambda m:section,text,flags=re.S)
    assert n==1,n
    text=text.replace('>预训练跟踪器替换方案</a>','>TWIST已运行结果与双机器人视频</a>')
    target.write_bytes(text.encode('utf-8'))
    print('TWIST_EXISTING_REPORT_UPDATED')

if __name__=='__main__':main()
