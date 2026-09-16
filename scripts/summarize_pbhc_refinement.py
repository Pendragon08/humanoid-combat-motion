"""Post-hoc descriptive diagnostics; never replaces the preregistered strict score."""
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from combat.twist_tracking import ROOT, scene

BASE = ROOT / 'runs/pbhc_refinement_pilot_20260916'

def main():
    protocol = json.loads((BASE/'protocol.json').read_text())
    quality = {(x['name'], x['arm']): x for x in json.loads((BASE/'reference_quality.json').read_text())}
    model, _ = scene(1)
    data = mujoco.MjData(model)
    names = ['left_wrist_roll_link', 'right_wrist_roll_link', 'left_ankle_roll_link', 'right_ankle_roll_link']
    bids = [model.body('a0_'+n).id for n in names]
    def local_fk(q):
        pos=[]
        for row in q:
            data.qpos[:]=row
            mujoco.mj_forward(model,data)
            pos.append(data.xpos[bids].copy())
        relative=np.asarray(pos)-q[:,:3,None].transpose(0,2,1)
        rot=Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix()
        return np.einsum('tji,tkj->tki',rot,relative)
    rows=[]
    for case in protocol['cases']:
        folder=BASE/'results'/case['id']
        metrics=json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'rollout.npz') as z:
            actual=z['qpos'][:,0].copy(); ref=z['reference'][:,0].copy()
        assert np.isfinite(actual).all() and len(actual)==metrics['frames']
        p=local_fk(actual); r=local_fk(ref)
        times=(np.arange(len(p))+1)*.02
        lo,hi=protocol['source_windows'][case['name']]
        active=(times>=lo)&(times<=hi)
        rv=np.linalg.norm(savgol_filter(r,11,3,deriv=1,delta=.02,axis=0),axis=-1)
        pv=np.linalg.norm(savgol_filter(p,11,3,deriv=1,delta=.02,axis=0),axis=-1)
        candidates=[0,1] if 'punch' in case['name'] else [2,3]
        # Selection depends ONLY on reference motion, never executed performance.
        eff=max(candidates,key=lambda e:np.quantile(rv[active,e],.95))
        start=np.flatnonzero(active)[0]
        rs=float(np.quantile(rv[active,eff],.95)); ps=float(np.quantile(pv[active,eff],.95))
        ra=float(np.linalg.norm(r[active,eff]-r[start,eff],axis=-1).max())
        pa=float(np.linalg.norm(p[active,eff]-p[start,eff],axis=-1).max())
        row={'name':case['name'],'arm':case['arm'],'strict':metrics['strict_complete'][0],
             'no_fall':metrics['no_fall_complete'][0],'first_fall_s':metrics['first_fall_s'][0],
             'strict_prefix_s':metrics['strict_prefix_s'][0],
             'joint_mae_rad':metrics['joint_mae_rad'][0],'root_error_m':metrics['root_error_m'][0],
             'body_error_m':metrics['tracked_body_error_m'][0],
             'selected_effector':names[eff],
             'active_effector_local_error_m':float(np.linalg.norm(p[active,eff]-r[active,eff],axis=-1).mean()),
             'reference_p95_local_speed_m_s':rs,'actual_p95_local_speed_m_s':ps,
             'p95_local_speed_ratio':ps/rs,'reference_local_excursion_m':ra,
             'actual_local_excursion_m':pa,'local_excursion_ratio':pa/ra,
             'final_root_error_m':float(np.linalg.norm(actual[-1,:3]-ref[-1,:3])),
             'reference_support_height_mae_m':quality[(case['name'],case['arm'])]['known_support_height_mae_m'],
             'reference_max_height_change_m':quality[(case['name'],case['arm'])]['maximum_height_change_m']}
        rows.append(row)
    aggregate=[]
    for arm in protocol['arms']:
        group=[x for x in rows if x['arm']==arm]
        aggregate.append({'arm':arm,'n':len(group),'no_fall':sum(x['no_fall'] for x in group),
            'strict':sum(x['strict'] for x in group),
            **{key:float(np.mean([x[key] for x in group])) for key in [
                'strict_prefix_s','joint_mae_rad','root_error_m','body_error_m',
                'active_effector_local_error_m','p95_local_speed_ratio','local_excursion_ratio',
                'reference_support_height_mae_m']}})
    output={'protocol':'Four development sources, frozen TWIST, one deterministic rollout per arm/source; no training or statistical generalization claim',
            'metric_notes':['Strict score and fall thresholds unchanged.',
                'Local speed/excursion diagnostics added after rollout; descriptive only, not a preregistered success criterion.',
                'Root-relative rotating-frame coordinates exclude global translation/rotation; always report global errors beside them.',
                '11-frame order3 Savitzky-Golay derivative; p95 active speed; effector selected by reference speed.',
                'Active intervals are coarse contact-sheet reviews, not expert contact annotations.',
                'Reference contact labels are heuristic; support-height error is NOT measured contact-force accuracy.'],
            'rows':rows,'aggregate':aggregate}
    (BASE/'summary.json').write_text(json.dumps(output,indent=2))
    with (BASE/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
