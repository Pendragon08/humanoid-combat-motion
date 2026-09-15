"""Audit geometry, deformation, speed, stance sliding and actual tracking."""
import json
import hashlib
from pathlib import Path
import shutil
import numpy as np
from combat.twist_tracking import ROOT

def additional(case):
    geo=np.load(case/'refinement/geometry_audit.npz');raw=geo['original_qpos'];fixed=geo['corrected_qpos']
    jids=np.r_[7:32,39:64];v0=np.diff(raw[:,jids],axis=0)*50;v1=np.diff(fixed[:,jids],axis=0)*50
    slide={};distance={}
    for name,folder in [('original',case/'refinement/original_export'),('corrected',case/'refinement')]:
        refs=[np.load(folder/f'actor{i}.npz') for i in range(2)]
        footpos=[]
        for z in refs:
            names=list(z['body_names']);footpos.extend([z['body_pos_w'][:,names.index(f'{side}_ankle_roll_link')] for side in ('left','right')])
        footpos=np.stack(footpos,axis=1);speeds=np.linalg.norm(np.diff(footpos[:,:,:2],axis=0),axis=2)*50
        mask=geo['support_mask'][1:]&geo['support_mask'][:-1]
        slide[name]=float(speeds[mask].mean()) if mask.any() else None
        distances=[]
        for i in range(2):
            me,other=refs[i],refs[1-i];mn=list(me['body_names']);on=list(other['body_names'])
            torso=other['body_pos_w'][:,on.index('torso_link')]
            for side in ('left','right'):
                hand=me['body_pos_w'][:,mn.index(side+'_wrist_roll_link')]
                distances.append(np.linalg.norm(hand-torso,axis=1))
        distance[name]=np.array(distances)
    return {'max_joint_speed_before_rad_s':float(np.abs(v0).max()),'max_joint_speed_after_rad_s':float(np.abs(v1).max()),
        'p95_joint_speed_before_rad_s':float(np.quantile(np.abs(v0),.95)), 'p95_joint_speed_after_rad_s':float(np.quantile(np.abs(v1),.95)),
        'stance_xy_speed_m_s':slide,
        'wrist_opponent_torso_distance_mae_change_m':float(np.abs(distance['corrected']-distance['original']).mean()),
        'proxy_note':'Stance mask heuristic. Wrist/torso distance is not an attack or contact label.'}

def main():
    root=ROOT/'runs/paired_refinement_v2';out=ROOT/'records/paired_refinement_v2/results';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for arm in ('generic','adapted'):
        for i in range(24):
            name=f'{arm}_{i:03}';case=root/name
            paths={'refinement':case/'refinement/refinement.json', 'original_export':case/'original_export/metrics.json','corrected':case/'corrected/metrics.json'}
            if any(not p.exists() for p in paths.values()):raise RuntimeError('Missing completed case: '+name)
            row={'case':name,**{k:json.loads(p.read_text()) for k,p in paths.items()},'additional':additional(case)}
            assert row['refinement']['code_sha256']==hashlib.sha256((ROOT/'combat/paired_refinement.py').read_bytes()).hexdigest()
            assert row['original_export']['code_sha256']==row['corrected']['code_sha256']
            dest=out/name;dest.mkdir(exist_ok=True)
            for key,p in paths.items():shutil.copy2(p,dest/(key+'.json'))
            (dest/'additional.json').write_text(json.dumps(row['additional'],indent=2));rows.append(row)
    groups={}
    for arm in ('generic','adapted','all'):
        items=[r for r in rows if arm=='all' or r['case'].startswith(arm)]
        total=sum(r['refinement']['frames'] for r in items)
        groups[arm]={'cases':len(items),'frames':total,
            'frames_over_1cm_before':sum(r['refinement']['frames_over_1cm_before'] for r in items),
            'frames_over_1cm_after':sum(r['refinement']['frames_over_1cm_after'] for r in items),
            'joint_change_mean_rad':float(np.mean([r['refinement']['mean_joint_change_rad'] for r in items])),
            'mean_pair_relation_max_change_m':float(np.mean([r['refinement']['max_pair_root_relation_change_m'] for r in items])),
            'cases_speed_max_increased':sum(r['additional']['max_joint_speed_after_rad_s']>r['additional']['max_joint_speed_before_rad_s'] for r in items),
            'stance_xy_speed_before_m_s':float(np.mean([r['additional']['stance_xy_speed_m_s']['original'] for r in items if r['additional']['stance_xy_speed_m_s']['original'] is not None])),
            'stance_xy_speed_after_m_s':float(np.mean([r['additional']['stance_xy_speed_m_s']['corrected'] for r in items if r['additional']['stance_xy_speed_m_s']['corrected'] is not None]))}
        for variant in ('original_export','corrected'):
            groups[arm][variant]={'strict_complete':sum(all(r[variant]['strict_complete']) for r in items),
                'pair_no_fall_complete':sum(all(r[variant]['no_fall_complete']) for r in items),
                'actor_no_fall_complete':sum(sum(r[variant]['no_fall_complete']) for r in items),
                'mean_min_pair_strict_prefix_s':float(np.mean([min(r[variant]['strict_prefix_s']) for r in items]))}
    result={'groups':groups,'results':rows,'claim':'Development diagnostic, not held-out performance or synthetic-data utility.',
        'matching':'Same exporter/differentiation, controller weights, simulation, timing and thresholds. Altered target poses imply strict tracking alone does not measure preservation of original combat intent.'}
    (ROOT/'records/paired_refinement_v2/summary.json').write_text(json.dumps(result,indent=2));print('PAIRED_REFINEMENT_SUMMARY',json.dumps(groups))

if __name__=='__main__':main()
