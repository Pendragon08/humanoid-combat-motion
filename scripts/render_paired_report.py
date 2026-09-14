"""Self-contained visual inspection of generated human motions; no force claims."""
import argparse
import hashlib
import html as html_module
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.run/'manifest.json').read_text(encoding='utf-8'))
    items=[]
    for r in manifest['samples']:
        f=a.run/(r['sample_id']+'.npz')
        if hashlib.sha256(f.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Sample hash mismatch')
        z=np.load(f,allow_pickle=False);x=z['positions'];fps=float(z['fps'])
        if x.shape[1:]!=(2,22,3) or not np.isfinite(x).all():raise ValueError('Invalid paired positions')
        feet=x[:,:,[7,8,10,11]]
        stats={'frames':len(x),'fps':fps,'minimum_foot_y_m':round(float(feet[...,1].min()),4),
               'p95_joint_speed_m_s':round(float(np.quantile(np.linalg.norm(np.diff(x,axis=0)*fps,axis=-1),.95)),4),
               'minimum_root_distance_m':round(float(np.linalg.norm(x[:,0,0]-x[:,1,0],axis=-1).min()),4)}
        items.append({'text':r['text'],'positions':np.round(x,4).tolist(),'stats':stats})
    payload=json.dumps(items,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双人格斗动作生成：实际运行结果</title><style>
body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f2f5f8;color:#182c40;margin:0}main{max-width:1100px;margin:auto;padding:30px}h1{font-size:29px}section{background:white;padding:22px;margin:18px 0;border-radius:14px}.notice{border-left:5px solid #e5a12b}button,select,input{font:inherit}canvas{width:100%;height:480px;background:#f5f8fa;touch-action:none}select{max-width:100%;padding:8px}button{padding:5px 18px}#time{width:60%}.legend{color:#526a7b}pre{white-space:pre-wrap;font-size:14px}.tag{color:#186e5a;font-weight:bold}a{color:#087a9b}@media print{button,input{display:none}section{break-inside:avoid}}</style>
<main><h1>双人格斗动作生成：实际运行结果</h1>
<section class="notice"><b>当前已验证：官方模型加载与双人动作生成。</b><br>蓝色和橙色分别为两个人的骨架，编号并不自动代表攻击者和防御者。下方动画是服务器实际生成的150帧、5秒人体动作，不是机器人实验录像。尚不能据此认定接触正确、机器人可执行或下游学习有效。</section>
<section><span class="tag">TIMotion · 官方 Transformer · 无 LPA · seed 0</span><p>POSTPROCESS_NOTE</p><p>选择提示词（Text prompt，文字动作条件）：</p><select id="choice"></select><p id="translation"></p>
<canvas id="view" width="1000" height="480"></canvas><p><button id="toggle">暂停</button> <input id="time" type="range" min="0" max="149" value="0"> <span id="frame"></span></p><p class="legend">拖动画面旋转视角。灰色为地面网格；骨架重叠不代表真实物理接触。</p><pre id="stats"></pre></section>
<section><h2>如何理解这些结果</h2><p>Reference Motion（参考动作）是控制器要跟随的目标。Retargeting（动作重定向）将人体运动转换为机器人关节目标。Motion Tracking（动作跟踪）需要机器人在物理仿真中用控制策略实际执行。三个阶段必须分别验证。</p><p>本页记录生成基线，不能替代官方 FID（生成分布质量）、接触成功率或机器人跟踪实验。双人共享同一个坐标系，未分别移动到原点；否则会破坏双方相对距离。</p><p>后续验证：生成动作 → 接触约束 → 机器人重定向 → 学习式跟踪 → 相同预算下比较真实与扩充数据。</p><p><a href="https://github.com/AIGC-Explorer/TIMotion">TIMotion 官方代码</a> · <a href="https://github.com/Pendragon08/humanoid-combat-motion">本项目代码与记录</a></p></section>
<script>const samples=PAYLOAD;
const cn=['拳击练习：一人出拳，另一人闪避。','武术练习：一人踢腿，另一人后退并格挡。','一人出拳，另一人格挡后反击。','两人拳击对练，移动并防御来拳。'];
const chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]];
const c=document.getElementById('view'),ctx=c.getContext('2d'),choice=document.getElementById('choice'),slider=document.getElementById('time');
samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.text;choice.appendChild(o)});
let frame=0,playing=true,yaw=.65,drag=false,lastX=0,last=0;
function selected(){return samples[+choice.value]};
function project(p){let x=p[0]*Math.cos(yaw)+p[2]*Math.sin(yaw),z=-p[0]*Math.sin(yaw)+p[2]*Math.cos(yaw);return [500+x*125,390-p[1]*125+z*32]}
function line(a,b,color,width){a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
function draw(){ctx.clearRect(0,0,c.width,c.height);for(let i=-4;i<=4;i++){line([i,0,-4],[i,0,4],'#d9e2e9',1);line([-4,0,i],[4,0,i],'#d9e2e9',1)}
const s=selected(),x=s.positions[frame];for(let a=0;a<2;a++){const color=a?'#ed8244':'#167bac';chains.forEach(ch=>{for(let j=1;j<ch.length;j++)line(x[a][ch[j-1]],x[a][ch[j]],color,4)});x[a].forEach(p=>{let v=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...v,3,0,Math.PI*2);ctx.fill()})}
slider.value=frame;document.getElementById('frame').textContent=`${frame+1} / ${s.positions.length} 帧`;}
function info(){const s=selected();document.getElementById('translation').textContent=cn[+choice.value]||'';document.getElementById('stats').textContent=`实际输出：${s.stats.frames} 帧，${s.stats.fps} 帧/秒\n脚部最低高度：${s.stats.minimum_foot_y_m} m（几何诊断）\n关节速度第95百分位：${s.stats.p95_joint_speed_m_s} m/s\n双方根节点最小距离：${s.stats.minimum_root_distance_m} m\n接触成功率与机器人执行成功率：尚未评估`;}
choice.onchange=()=>{frame=0;info();draw()};slider.oninput=()=>{frame=+slider.value;playing=false;document.getElementById('toggle').textContent='播放';draw()};
document.getElementById('toggle').onclick=e=>{playing=!playing;e.target.textContent=playing?'暂停':'播放'};
c.onpointerdown=e=>{drag=true;lastX=e.clientX;c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=false;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-lastX)*.01;lastX=e.clientX;draw()}};
function tick(t){if(playing&&t-last>1000/30){frame=(frame+1)%selected().positions.length;draw();last=t}requestAnimationFrame(tick)}info();draw();requestAnimationFrame(tick);
</script></main></html>'''.replace('PAYLOAD',payload)
    note=('已做人物编号连续性校正：保持每帧两具骨架的位置与姿态，只重新对应人物编号。'
          '这是工程后处理，不是接触优化或机器人动力学约束；原始输出另行保留。' if manifest.get('postprocess') else
          '原始模型输出，未做人物编号连续性校正。部分帧存在编号互换，不能直接用作机器人参考轨迹。')
    html=html.replace('POSTPROCESS_NOTE',html_module.escape(note))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(html,encoding='utf-8')
    a.output.with_suffix('.metrics.json').write_text(json.dumps([x['stats'] for x in items],indent=2),encoding='utf-8')
    print(a.output)


if __name__=='__main__':main()
