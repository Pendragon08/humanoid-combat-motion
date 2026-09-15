"""Collect all predeclared outcomes, including failures, into existing report."""
import hashlib
import json
from pathlib import Path
import shutil
from combat.twist_tracking import ROOT

def main():
    src=ROOT/'runs/twist_suite_v2';out=ROOT/'records/twist/results';out.mkdir(parents=True,exist_ok=True)
    paths=sorted(src.glob('*/metrics.json'))
    if len(paths)!=57:raise RuntimeError(f'Incomplete suite: {len(paths)}/57')
    rows=[]
    for p in paths:
        row=json.loads(p.read_text());row['case']=p.parent.name
        assert row['code_sha256']==json.loads(paths[0].read_text())['code_sha256'],'Mixed code versions'
        dest=out/p.parent.name;dest.mkdir(exist_ok=True)
        shutil.copy2(p,dest/'metrics.json')
        rollout=p.parent/'rollout.npz'
        row['rollout_sha256']=hashlib.sha256(rollout.read_bytes()).hexdigest()
        rows.append(row)
    groups={}
    for prefix in ('kungfu_train20','kungfu_test4','dual_generic','dual_adapted'):
        selected=[r for r in rows if r['case'].startswith(prefix)]
        kinds={k:sum(any(c['kind']==k and c['penetration_m']>.01 for c in r['initial_penetrating_contacts']) for r in selected) for k in ('ground','self','interrobot')}
        groups[prefix]={'cases':len(selected),'strict_complete_cases':sum(all(r['strict_complete']) for r in selected),
            'no_fall_complete_cases':sum(all(r['no_fall_complete']) for r in selected),
            'actor_strict_complete':sum(sum(r['strict_complete']) for r in selected),
            'actor_no_fall_complete':sum(sum(r['no_fall_complete']) for r in selected),
            'cases_with_initial_penetration_over_1cm':sum(r['initial_max_penetration_m']>.01 for r in selected),
            'initial_penetration_over_1cm_by_kind_top20_contacts':kinds,
            'cases_with_interrobot_contact':sum(r['interrobot_contact_control_frames']>0 for r in selected)}
    summary={'source':'runs/twist_suite_v2','total_cases':57,'groups':groups,'results':rows,
        'criterion':'Every frame joint MAE<0.3 rad, root error<0.15m, eight-body world position mean error<0.15m, pelvis height>=0.35m.',
        'comparison_limit':'Different simulator, pretrained controller, nominal start and 25 retained DOFs; no algorithm-only improvement claim.'}
    (ROOT/'records/twist/summary.json').write_text(json.dumps(summary,indent=2))
    print('TWIST_SUITE_SUMMARY',json.dumps(groups))

if __name__=='__main__':main()
