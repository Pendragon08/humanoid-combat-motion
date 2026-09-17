"""Compare raw and prior candidates on the entire original-time trajectory."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from audit_pbhc_reference_kinematics import fk
from analyze_pbhc_support_diagnostics import sole_minimum

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_quality_gate_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def derivative(x,order):return np.diff(x,n=order,axis=0)*50**order
def stats(x):
    values=np.abs(x)
    return dict(max=float(values.max()),p99=float(np.percentile(values,99)),rms=float(np.sqrt(np.mean(values**2))))

def geometry(q,names):
    tf=fk(q,names)
    feet=np.stack([tf[f'{s}_ankle_roll_link'][:,:3,3] for s in ['left','right']],axis=1)
    low=np.stack([sole_minimum(tf,s) for s in ['left','right']],axis=1)
    return feet,low

def main():
    REC.mkdir(parents=True,exist_ok=True);target=REC/'temporal_audit.json'
    if target.exists():raise FileExistsError(target)
    prior=json.loads((ROOT/'records/pbhc_support_repair_20260917/protocol.json').read_text())
    inventory=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())['motions']
    windows=json.loads((ROOT/'runs/pbhc_refinement_pilot_20260916/protocol.json').read_text())['source_windows']
    results=[]
    for name in sorted(set(r['name'] for r in prior['records'])):
        source=ROOT/'runs/pbhc_beyondmimic_20260916/references_full'/f'sample_{name}_actor0.npz'
        with np.load(source,allow_pickle=False) as z:
            names=list(z['joint_names']);raw=np.c_[z['body_pos_w'][:,0],z['body_quat_w'][:,0],z['joint_pos']].astype(float)
            reference_joint_velocity=z['joint_vel'].copy()
        case=next(c for c in inventory if c['name']==name)
        contact=np.asarray(next(iter(joblib.load(case['source_path']).values()))['contact_mask'])
        idx=np.minimum(np.rint(np.arange(len(raw))/50*case['source_fps']).astype(int),len(contact)-1);contact=contact[idx]
        changes=np.any(contact[1:]!=contact[:-1],axis=1)
        change_frames=np.flatnonzero(changes)+1
        phase_indices=[int(round(s*50)) for s in windows[name]]
        candidates=[('raw',raw,source)]
        for row in [r for r in prior['records'] if r['name']==name]:
            with np.load(row['input'],allow_pickle=False) as z:candidates.append((row['arm'],z['qpos'],Path(row['input'])))
        feet0,low0=geometry(raw,names)
        for arm,q,path in candidates:
            feet,low=geometry(q,names);delta=q[:,7:]-raw[:,7:]
            dq=derivative(q[:,7:],1);ddq=derivative(q[:,7:],2)
            dz=q[:,2]-raw[:,2]
            band=np.zeros(len(q),bool)
            for ix in phase_indices:band[max(0,ix-5):min(len(q),ix+6)]=True
            leg=[i for i,n in enumerate(names) if any(k in n for k in ['hip_','knee_','ankle_'])]
            peak=np.unravel_index(np.abs(ddq[:,leg]).argmax(),ddq[:,leg].shape)
            row=dict(name=name,arm=arm,input_sha256=sha(path),source_sha256=sha(source),frames=len(q),
                joint_velocity_rad_s=stats(dq),joint_acceleration_rad_s2=stats(ddq),joint_jerk_rad_s3=stats(derivative(q[:,7:],3)),
                root_velocity_m_s=stats(derivative(q[:,:3],1)),root_acceleration_m_s2=stats(derivative(q[:,:3],2)),
                added_joint_velocity_rad_s=stats(derivative(delta,1)),added_joint_acceleration_rad_s2=stats(derivative(delta,2)),
                added_root_z_velocity_m_s=stats(derivative(dz,1)),added_root_z_acceleration_m_s2=stats(derivative(dz,2)),
                full_penetration_max_m=float(np.maximum(-low,0).max()),full_penetration_mean_m=float(np.maximum(-low,0).mean()),
                framewise_penetration_worsened_fraction=float((np.maximum(-low,0)>np.maximum(-low0,0)+1e-6).mean()),
                source_known_contact_gap_mean_m=float(np.abs(low[np.isclose(contact,1)]).mean()),
                source_unknown_contact_fraction=float(np.isclose(contact,.5).mean()),
                source_label_change_frames=change_frames.tolist(),phase_boundary_frames=phase_indices,
                phase_boundary_added_acceleration_max_rad_s2=float(np.abs(derivative(delta,2))[band[1:-1]].max()),
                peak_leg_acceleration=dict(joint=names[leg[peak[1]]],center_time_s=(peak[0]+1)/50,value_rad_s2=float(ddq[peak[0],leg[peak[1]]])),
                maximum_foot_xy_change_m=float(np.linalg.norm(feet[:,:,:2]-feet0[:,:,:2],axis=-1).max()))
            if arm=='raw':
                row['stored_joint_velocity_vs_central_difference_max_rad_s']=float(np.max(np.abs(reference_joint_velocity-np.gradient(raw[:,7:],.02,axis=0))))
            results.append(row)
    output=dict(scope='Source-relative geometry and finite differences, all frames; no controller results used',
        derivative_definition='Forward differences at dt=.02s; order2 centered on interior frames. Source coordinate convention unchanged.',
        caveats=['Contact labels are upstream heuristics, unknown .5 excluded from known support statistics',
                 'URDF capsule lower envelope is geometric proxy, not measured contact',
                 'Finite-difference acceleration/jerk are data regularity metrics, not torque feasibility certificates'],
        script_sha256=sha(Path(__file__)),rows=results)
    target.write_text(json.dumps(output,indent=2))
    for r in results:
        print(r['name'],r['arm'],'vel_max',round(r['joint_velocity_rad_s']['max'],3),
              'acc_max',round(r['joint_acceleration_rad_s2']['max'],3),
              'added_acc',round(r['added_joint_acceleration_rad_s2']['max'],3),
              'boundary_added_acc',round(r['phase_boundary_added_acceleration_max_rad_s2'],3),
              'penetration_max',round(r['full_penetration_max_m'],4),flush=True)

if __name__=='__main__':main()
