"""Check completed-run inventory and local HTML links without inventing results."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlsplit

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];snapshot=root/'records/paired_v1/completed_snapshot'
expected={'training_references':('g1_tracking_eval_multiref_train20_*/metrics.json',20),
          'heldout_references':('g1_tracking_eval_multiref_test4_*/metrics.json',4),
          'contact_refined_pairs':('dual_g1_contact_*_multiref999/metrics.json',20)}
counts={}
for key,(pattern,count) in expected.items():
    paths=list(snapshot.glob(pattern));counts[key]=len(paths)
    if len(paths)!=count:raise ValueError(f'Incomplete {key}: {len(paths)}/{count}')
    for path in paths:
        r=json.loads(path.read_text());assert r['horizon_steps']>0 and 0<=r['completion_fraction']<=1
attention=json.loads((snapshot/'actor_reaction_crossattention_test_seed123/evaluation.json').read_text())
assert attention['history_only'] and attention['cross_attention'] and not attention['actor_future_as_input']
assert len(attention['cases'])==20
report=root/'reports/experiment_progress_20260914.html';text=report.read_text(encoding='utf-8')
if re.search(r'__[A-Z_]+__',text):raise ValueError('Unfilled HTML template')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        if 'id' in data:self.ids.append(data['id'])
        for key in ('href','src','poster'):
            if key in data:self.paths.append(data[key])
links=Links();links.feed(text)
if len(set(links.ids))!=len(links.ids):raise ValueError('Duplicate HTML IDs')
local=[]
for link in links.paths:
    url=urlsplit(link)
    if not url.scheme and url.path:
        path=report.parent/unquote(url.path)
        if not path.is_file():raise FileNotFoundError(path)
        local.append(str(path.relative_to(root)))
    if not url.path and url.fragment and url.fragment not in links.ids:raise ValueError('Broken section link')
result={'completed_inventory':counts,'cross_attention_test_cases':20,'local_html_assets_checked':sorted(set(local)),
        'full_research_plan_complete':False,'synthetic_data_utility_validated':False,
        'remaining':['validated physical critic','candidate reranking','equal-budget three-seed downstream data utility'],
        'reason':'multi-reference tracking coverage is low; contact-constrained human generation does not yet give a validated robot-executable dataset'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
