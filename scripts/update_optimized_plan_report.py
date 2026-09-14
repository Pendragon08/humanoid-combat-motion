"""Embed the bounded plan section in the existing HTML; no third-party renderer."""
from pathlib import Path
import html
import re

root = Path(__file__).resolve().parents[1]
protocol = (root / 'docs/step7_protocol.md').read_text(encoding='utf-8')
text = protocol.split('<!-- OPTIMIZED-PLAN-BEGIN -->', 1)[1].split('<!-- OPTIMIZED-PLAN-END -->', 1)[0]

def inline(value):
    value = html.escape(value)
    value = re.sub(r'\[([^\]]+)\]\((https://[^\s)]+)\)', r'<a href="\2">\1</a>', value)
    value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)

parts = []
lines = text.strip().splitlines()
i = 0
details_open = False
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith('### '):
        if details_open:
            parts.append('</details>')
        parts.append('<details open><summary><strong>' + inline(line[4:]) + '</strong></summary>')
        details_open = True
    elif line.startswith('## '):
        parts.append('<h2>' + inline(line[3:]) + '</h2>')
    elif line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            cells = [x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', x) for x in cells):
                rows.append(cells)
            i += 1
        parts.append('<div style="overflow-x:auto"><table>')
        for n, cells in enumerate(rows):
            tag = 'th' if n == 0 else 'td'
            parts.append('<tr>' + ''.join(f'<{tag}>{inline(x)}</{tag}>' for x in cells) + '</tr>')
        parts.append('</table></div>')
        continue
    else:
        parts.append('<p>' + inline(line) + '</p>')
    i += 1
if details_open:
    parts.append('</details>')
start, end = '<!-- OPTIMIZED-PLAN-BEGIN -->', '<!-- OPTIMIZED-PLAN-END -->'
section = start + '<section id="optimized-plan">' + ''.join(parts) + '</section>' + end
page = root / 'reports/experiment_progress_20260914.html'
source = page.read_text(encoding='utf-8')
if start in source:
    source = source[:source.index(start)] + section + source[source.index(end) + len(end):]
else:
    marker = '<!-- STEP7-END -->'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + section, 1)
if '<a href="#optimized-plan">' not in source:
    source = source.replace('<nav>', '<nav><a href="#optimized-plan">9月15日优化方案</a> · ', 1)
source = source.replace('2026-09-14 增量记录 · 原项目继续实施 · 所有数值来自服务器保存的实验输出',
                        '2026-09-15 更新 · 既有结果与待实施方案分别标注 · 验收门槛不是实验成绩')
page.write_bytes(source.encode('utf-8'))
print('Existing report updated; optimized plan is proposed, not executed.')
