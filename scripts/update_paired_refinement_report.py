"""Update existing HTML with the complete paired correction experiment."""
import html
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]

def main():
    s=json.loads((ROOT/'records/paired_refinement_v2/summary.json').read_text(encoding='utf-8'));g=s['groups']['all']
    rows=''
    for key,title in [('generic','24组通用生成'),('adapted','24组适配生成'),('all','合计48组')]:
        group=s['groups'][key];a=group['original_export'];b=group['corrected'];n=group['cases']
        rows+=f"<tr><td>{title}</td><td>{a['strict_complete']}/{n} → {b['strict_complete']}/{n}</td><td>{a['pair_no_fall_complete']}/{n} → {b['pair_no_fall_complete']}/{n}</td><td>{group['frames_over_1cm_before']}/{group['frames']} → {group['frames_over_1cm_after']}/{group['frames']}</td></tr>"
    detail=''
    for r in s['results']:
        a=r['original_export'];b=r['corrected'];ref=r['refinement'];ad=r['additional']
        detail+=f"<tr><td>{r['case']}</td><td>{all(a['no_fall_complete'])} → {all(b['no_fall_complete'])}</td><td>{ref['frames_over_1cm_before']} → {ref['frames_over_1cm_after']}</td><td>{ref['mean_joint_change_rad']:.3f}</td><td>{ad['max_joint_speed_before_rad_s']:.1f} → {ad['max_joint_speed_after_rad_s']:.1f}</td><td>{ad['wrist_opponent_torso_distance_mae_change_m']:.3f}</td></tr>"
    block=f'''<!-- PAIRED-REFINEMENT-BEGIN -->
<section id="paired-refinement"><h2>下一步已执行：双人参考动作的几何与支撑脚修正</h2>
<p><b>本轮完成48组几何优化、96次真实控制器执行。</b>保留上一轮所有原始输入和结果。使用同一个TWIST预训练策略、同一个MuJoCo机器人及碰撞几何，时间长度、50Hz参考帧率、起始和成功判定规则相同。两台机器人始终在同一物理场景中。</p>
<p>上一版位置重定向主要拟合人体关节位置，没有加入机器人碰撞体和支撑脚约束。人体脚踝的位置不等于机器人脚底应当接触地面的位置，肢体形状也不同。于是“能转换为关节角度”并不意味着姿势无穿插。</p>
<h3>这次具体做了什么？</h3>
<ol><li>在TWIST实际使用的机器人模型中读取地面、自身和双方的碰撞。使用接触法线及Jacobian（关节变化如何影响接触点位置的关系矩阵）估计修正方向。</li>
<li>用Sequential QP（逐次二次规划，即每次求一个小幅度的约束修正）同时考虑关节限位、原姿势保真、修正量连续性和碰撞。通过实际几何重新计算代价，避免只相信局部近似。</li>
<li>根据脚的相对高度和水平速度，估计支撑脚；约束脚底高度以及支撑期间的水平位置。这是启发式估计，不是真实接触标签。</li>
<li>限制关节和根部改动，保留共同世界坐标和原时序。没有降低播放速度、用站立替换动作或把两人移开来消除碰撞。</li>
<li>保存修正残差和动作变化，再由原控制器实际执行；几何优化后仍有碰撞或跌倒的片段照实计入。</li></ol>
<p>R0＝原始姿势重新导出；R1＝修正姿势用同样方法导出。两组使用相同的正向运动学和速度重算方式，避免把格式转换当成优化收益。以前原始NPZ的直接运行保留作背景参考，不替代本轮R0对照。</p>
<p><b>首帧几何对照：</b>左为原始参考，右为修正参考。这里只检查形状和穿插，尚未运行控制器，不能把这张图当成机器人稳定执行。</p><img src="media/paired_geometry_000.jpg" alt="双人参考动作修正前后首帧几何对照，不代表动力学执行" loading="lazy">
<table><tr><th>测试组</th><th>双人严格成功：R0→R1</th><th>两台全程未跌倒：R0→R1</th><th>几何穿插&gt;1cm帧数：前→后</th></tr>{rows}</table>
<p>几何穿插按每帧原始/修正参考姿势的碰撞检测统计；跟踪成功和跌倒来自实际控制器仿真，二者不是同一指标。双人严格成功要求两台每帧满足关节、根部及身体位置误差门槛，而不是只要求站着。</p>
<h3>动作改动和副作用</h3>
<ul><li>按片段平均的关节改动：{g['joint_change_mean_rad']:.3f}弧度。单关节设置0.65弧度变化范围，求解器有数值容差；这不是语义不变证明。</li>
<li>每片段最大双方根部相对位置变化的平均值：{g['mean_pair_relation_max_change_m']:.3f}米。共同坐标没有重置，但约束修正仍会改变双方相对几何。</li>
<li>在估计支撑区间内，参考脚部水平速度：{g['stance_xy_speed_before_m_s']:.3f} → {g['stance_xy_speed_after_m_s']:.3f}米/秒。这是参考动作的滑动代理指标，不是物理执行时滑动量。</li>
<li>{g['cases_speed_max_increased']}/48个片段的最大关节速度增加。当前没有硬性的全轨迹速度、加速度及力矩约束，不能将几何修正当作动力学可行性保证。</li></ul>
<p class="notice"><b>本轮仍是开发阶段的诊断。</b>减少穿插不等于动态稳定，也不等于正确攻击、格挡或反击。若严格成功仍为零，应把这一结果视为“仅靠局部几何后处理不足”的证据，不能宣布物理约束格斗生成已经成功。试运行使用过适配样本000，本轮48组不作为未见测试集。尚未训练新的模仿策略，因此没有合成数据提高学习性能的结论。</p>
<h3>真实双人执行：预先指定的样本000</h3>
<p>两个视频都包含同一物理场景中的两台机器人。一个是原始姿势经过统一导出后的执行，另一个是修正后执行。全部状态由策略和物理仿真产生，视频为保存状态重渲染。</p>
<p><b>R0：原始姿势、统一导出。</b></p><video controls preload="metadata" src="media/paired_refine_original_000.mp4"></video>
<p><b>R1：几何与支撑脚修正。</b>是否成功以表格为准，不能只凭是否看起来在运动判断。</p><video controls preload="metadata" src="media/paired_refine_corrected_000.mp4"></video>
<details><summary>48组逐项审计：保留所有失败和副作用</summary><table><tr><th>样本</th><th>双人未跌倒</th><th>穿插&gt;1cm帧</th><th>平均关节改动/rad</th><th>最大关节速度/rad·s⁻¹</th><th>手腕—对方躯干距离平均变化/m</th></tr>{detail}</table><p>手腕—躯干距离变化只检验交互几何变化，不表示已建立准确攻防接触标签。</p></details>
<h3>下一阶段应解决的具体问题</h3>
<p>把“先生成自由人体动作、再局部修补”升级为目标机器人约束下的轨迹优化：在生成或重定向阶段就约束脚底接触、姿态、速度与加速度，并通过实际跟踪反馈筛选。对双人动作还要保留攻击—反应时序及可达接触。先做短时、低冲击的双人格挡/闪避任务，确认可靠跟踪后再扩展高速碰撞；不能继续仅靠增加训练步数掩盖参考质量问题。</p>
<p>可核验记录：<a href="../records/paired_refinement_v2/summary.json">完整指标</a> · <a href="../docs/paired_refinement_execution.md">方法与执行说明</a> · <a href="../records/paired_refinement_v2/independent_witness.txt">独立数值见证</a>。接触法线约定依据<a href="https://mujoco.readthedocs.io/en/stable/computation/">MuJoCo官方文档</a>；未改动碰撞掩码。修复了跨帧初始修正量越过关节限位的问题，最终结果来自独立v2目录，未将未完成的v1批次混入。</p>
</section><!-- PAIRED-REFINEMENT-END -->'''
    path=ROOT/'reports/experiment_progress_20260914.html';text=path.read_text(encoding='utf-8')
    if '<!-- PAIRED-REFINEMENT-BEGIN -->' in text:
        text=re.sub(r'<!-- PAIRED-REFINEMENT-BEGIN -->.*?<!-- PAIRED-REFINEMENT-END -->',lambda m:block,text,flags=re.S)
    else:text=text.replace('<!-- TRACKER-REPLACEMENT-AUDIT-BEGIN -->',block+'\n<!-- TRACKER-REPLACEMENT-AUDIT-BEGIN -->',1)
    if 'href="#paired-refinement"' not in text:text=text.replace('<nav>','<nav><a href="#paired-refinement">双人几何修正新实验</a> · ',1)
    path.write_bytes(text.encode('utf-8'));print('PAIRED_REFINEMENT_REPORT_UPDATED')

if __name__=='__main__':main()
