"""Integrate the current data-focused protocol into the existing HTML report.

This updates documentation only. It does not launch or relabel experiments.
The small renderer intentionally handles the prose/table subset of our protocol.
"""
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEGIN = '<!-- DATA-CENTRIC-PROTOCOL-BEGIN -->'
END = '<!-- DATA-CENTRIC-PROTOCOL-END -->'


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)


def render_prose(source):
    lines, out, paragraph = source.splitlines(), [], []

    def flush():
        if paragraph:
            out.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush()
        elif line.startswith('|'):
            flush()
            table = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    table.append(cells)
                i += 1
            out.append('<div style="overflow-x:auto"><table>')
            for n, row in enumerate(table):
                tag = 'th' if n == 0 else 'td'
                out.append('<tr>'+''.join(f'<{tag}>{inline(c)}</{tag}>' for c in row)+'</tr>')
            out.append('</table></div>')
            continue
        elif line.startswith('#'):
            flush()
            match = re.fullmatch(r'(#{2,3}) (.+)', line)
            if not match:
                raise ValueError('Unsupported heading: '+line)
            level = len(match[1])+1
            out.append(f'<h{level}>{inline(match[2])}</h{level}>')
        elif line == '---':
            flush()
            out.append('<hr>')
        else:
            paragraph.append(line)
        i += 1
    flush()
    return '\n'.join(out)


def main():
    plan = json.loads((ROOT/'configs/data_utility_protocol_v2.json').read_text(encoding='utf-8'))
    if plan['status'] != 'planned_not_executed' or plan['launch_enabled']:
        raise ValueError('This report renderer is only for the unlaunched revised protocol')
    md = (ROOT/'docs/step7_protocol.md').read_text(encoding='utf-8')
    start = md.index('## 当前执行协议：以示范数据为核心')
    stop = md.index('## 历史协议与结果（保留供复现', start)
    prose = render_prose(md[start:stop])
    rows = ''.join('<tr><td>'+inline(a['id'])+'</td><td>'+inline(a['label_zh'])+'</td></tr>' for a in plan['arms'])
    section = f'''{BEGIN}
<section id="data-centric-protocol"><h2>当前研究主线：构建高质量格斗示范数据</h2>
<p class="notice"><b>2026-09-15范围修订，以下是新计划，尚未运行新实验。</b>研究重点是减少专业真人MoCap依赖的示范数据获取、生成和质量控制。Motion Tracking（动作跟踪）作为统一的下游学习验证工具；既有TWIST与几何修正实验保留为工程诊断。</p>
<p><b>课题：</b>{inline(plan['title_zh'])}。核心判断是“生成的数据是否更好，以及是否真正改善学习或减少真人采集量”。新协议优先于下方历史优化建议。</p>
<p><b>主流程：</b>MoCap／视频 → 来源审计 → 格斗动作生成 → 接触与物理约束 → 机器人参考数据 → 同一学习算法分别训练 → 独立测试与数据用量曲线。</p>
<table><tr><th>新协议组别</th><th>数据对照</th></tr>{rows}</table>
<p>各组同样的网络、奖励、预算和初始化，每组分别训练策略。进一步比较10/25/50/100%真人MoCap用量；不能用一个冻结控制器播放成功来代替数据效用实验。</p>
<p><b>现阶段边界：</b>KungFuAthlete种子来自视频恢复，不是传感器MoCap。可先做视频种子扩增；专业MoCap节省、独立多来源测试和完整双人格斗学习仍待验证。旧的3.39%／0.78%／0.78%是历史视频种子试验，不是新四组结果。</p>
<p><b>下一个交付：</b>带来源、角色、接触置信度、成本、质量指标及失败案例的数据清单。跟踪器只做有限接口校准；无需先让旧控制器执行全部候选。</p>
<details><summary>展开完整修订协议：主张、5个实验块、MoCap节省、统计与2×A6000预算</summary>{prose}</details>
<p><a href="../docs/step7_protocol.md">完整协议源文件</a> · <a href="../configs/data_utility_protocol_v2.json">机器可读计划（禁止自动启动）</a></p>
</section>
{END}'''
    target = ROOT/'reports/experiment_progress_20260914.html'
    page = target.read_text(encoding='utf-8')
    nav_start = page.index('<nav>')
    prefix = page[:nav_start]
    prefix, count = re.subn(
        r'<div class="notice">.*?</div>',
        '<div class="notice"><b>当前核心问题：生成数据能否补充真人MoCap并改善格斗模仿学习？</b>'
        '尚未获得肯定证据。已有生成、重定向与跟踪诊断结果保留；新方案以数据来源、质量、成本、独立训练效用和MoCap用量曲线为主线。</div>',
        prefix, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Missing report lead notice')
    page = prefix+page[nav_start:]
    if BEGIN in page:
        start = page.index(BEGIN)
        stop = page.index(END, start)+len(END)
        page = page[:start]+section+page[stop:]
    else:
        pos = page.index('</nav>')+len('</nav>')
        page = page[:pos]+'\n'+section+page[pos:]
    link = '<a href="#data-centric-protocol">当前数据研究主线</a> · '
    if link not in page:
        page = page.replace('<nav>', '<nav>'+link, 1)
    for anchor in ('paired-refinement', 'tracker-replacement', 'tracking-diagnostics-results', 'optimized-plan'):
        marker = f'<!-- DATA-SCOPE-HISTORY-{anchor} -->'
        if marker not in page:
            pattern = rf'(<section id="{anchor}">\s*<h2>.*?</h2>)'
            notice = marker+'<p class="notice">历史工程记录／当时方案。当前优先级已改为示范数据构建和训练效用验证，见<a href="#data-centric-protocol">当前协议</a>；本节不作为继续优化跟踪算法的任务指令。</p>'
            page, count = re.subn(pattern, lambda m: m[1]+notice, page, count=1, flags=re.S)
            if count != 1:
                raise ValueError('Missing historical section: '+anchor)
    target.write_bytes(page.replace('\r\n', '\n').encode('utf-8'))
    print('Updated existing report; new protocol planned, no experiments launched.')


if __name__ == '__main__':
    main()
