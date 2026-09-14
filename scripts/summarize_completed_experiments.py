"""Derive compact summaries only from existing run artifacts."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs'))
p.add_argument('--output',type=Path,required=True);a=p.parse_args();summary={}
for path in sorted(a.root.glob('g1_tracking_eval_*/metrics.json')):
    if 'primed' not in path.parent.name:continue
    r=json.loads(path.read_text());summary[path.parent.name]={k:r[k] for k in ('completion_fraction','horizon_steps','mean_alive_error','strict_tracking_pass_fraction') if k in r}
for path in sorted(a.root.glob('actor_reaction_*/evaluation.json')):
    r=json.loads(path.read_text());rows=r['cases'];out={k:r[k] for k in ('method','status','history_only','contact_success_fraction','maximum_bone_change_m','optimization_surface_gap_target_m') if k in r}
    if rows and 'arms' in rows[0]:
        out['arms']={arm:{'contact_success':statistics.mean(float(row['arms'][arm]['contact']['success']) for row in rows),
                         'defender_position_rmse_m':statistics.mean(row['arms'][arm]['defender_position_rmse_m'] for row in rows)} for arm in rows[0]['arms']}
    summary[path.parent.name]=dict(out,cases=len(rows))
for path in sorted(a.root.glob('multireference_*/TRAINING_COMPLETE.json')):
    summary[path.parent.name]=json.loads(path.read_text())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
