"""Final artifact checks; completion of a pilot is not physics validation."""
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote

root=Path(__file__).resolve().parents[1]
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
summary=read(root/'records/step7/summary.json')
assert summary['pilot_runs_complete'] and summary['completed_runs']==9
assert {(r['arm'],r['seed']) for r in summary['rows']}=={(a,s) for a in ('real','generic','contact') for s in (0,1,2)}
assert len({r['model_sha256'] for r in summary['rows']})==9,'Identical checkpoints across supposedly independent runs'
assert all(len(r['evaluations'])==4 for r in summary['rows'])
assert summary['full_step7_physics_pipeline_validated'] is False
assert summary['critic']['decision'].startswith('ABSTAIN')
index=read(root/'records/step7/snapshot/INDEX.json')
for row in index['files']:
    data=(root/row['saved']).read_bytes()
    if 'uncompressed_sha256' in row:
        data=gzip.decompress(data);expected=row['uncompressed_sha256']
    else:expected=row['sha256']
    assert hashlib.sha256(data).hexdigest()==expected,row['saved']

class ReportParser(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id'])
        for key in ('href','src','poster'):
            if key in attrs:self.links.append(attrs[key])
page=root/'reports/experiment_progress_20260914.html';parser=ReportParser();parser.feed(page.read_text(encoding='utf-8'))
assert len(parser.ids)==len(set(parser.ids)),'Duplicate section IDs'
for link in parser.links:
    if link.startswith('#'):assert link[1:] in parser.ids
    elif not link.startswith(('http:','https:','data:')):assert (page.parent/unquote(link.split('#')[0])).exists(),link
for name in ('step7_utility.mp4','step7_utility_preview.png','step7_utility_provenance.json'):
    assert (root/'reports/media'/name).stat().st_size>0
provenance=read(root/'reports/media/step7_utility_provenance.json')
for directory,expected in zip(provenance['source_evaluations'],provenance['source_rollout_sha256']):
    assert hashlib.sha256((root/directory/'rollout_env0.npz').read_bytes()).hexdigest()==expected
result={'complete_policy_runs':9,'reused_policy_runs':1,'new_policy_runs':8,'new_environment_steps':98304000,
        'evaluation_records':36,'different_checkpoint_hashes':9,'compressed_records_verified':len(index['files']),
        'html_ids_and_links_valid':True,'video_rollout_hashes_valid':True,
        'full_physics_critic_or_paired_combat_validated':False,'pilot_execution_complete':True}
(root/'records/step7/artifact_check.json').write_text(json.dumps(result,indent=2))
print('STEP7_ARTIFACT_CHECK',json.dumps(result),flush=True)
