"""Read-only failure timing, contact-force and reference-foot geometry analysis."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_pbhc_reference_kinematics import fk, tree, center_of_mass, xyz

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_support_diagnostic_20260917'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def sole_minimum(transforms,side):
    """Analytic capsule lower envelope matching cylinder-to-capsule intent.

    URDF-derived proxy, not a measured contact or USD geometry certification.
    Never interpret horizontal motion of this changing extremum as foot slip.
    """
    link=tree.find(f"link[@name='{side}_ankle_roll_link']")
    tf=transforms[link.get('name')];mins=[]
    for col in link.findall('collision'):
        cylinder=col.find('geometry/cylinder');assert cylinder is not None
        origin=col.find('origin');axis=Rotation.from_euler('xyz',xyz(origin,'rpy')).as_matrix()[:,2]
        world=tf[:,:3,:3]@xyz(origin)+tf[:,:3,3]
        axisworld=tf[:,:3,:3]@axis
        mins.append(world[:,2]-np.abs(axisworld[:,2])*float(cylinder.get('length'))/2-float(cylinder.get('radius')))
    return np.min(mins,axis=0)

def first(mask,fps=50):
    ids=np.flatnonzero(mask);return float((ids[0]+1)/fps) if len(ids) else None

def main():
    assert (REC/'COMPLETE.json').exists()
    target=REC/'support_analysis_v2.json'
    if target.exists():raise FileExistsError(target)
    manifest=json.loads((REC/'manifest.json').read_text())
    rows=[]
    for job in manifest['jobs']:
        folder=ROOT/job['output'];metrics=json.loads((folder/'metrics.json').read_text())
        old=json.loads((ROOT/job['old_output']/'metrics.json').read_text())
        with np.load(folder/'support_trace.npz',allow_pickle=False) as z,np.load(ROOT/job['reference'],allow_pickle=False) as ref:
            assert sha(folder/'support_trace.npz')==metrics['trace_sha256']
            assert sha(ROOT/job['reference'])==job['reference_sha256']
            bn=list(z['body_names']);jn=list(z['joint_names']);sn=list(z['sensor_body_names']);tn=list(z['termination_names'])
            assert list(ref['body_names'])==bn and list(ref['joint_names'])==jn
            valid=z['alive'].astype(bool);fps=float(z['fps']);times=np.arange(1,len(valid)+1)/fps
            actual=z['body_pos']-z['env_origins'][None,:,None,:]
            delta=actual-ref['body_pos_w'][1:,None]
            joint_abs=np.abs(z['joint_pos']-ref['joint_pos'][1:,None])
            feet=[bn.index(f'{s}_ankle_roll_link') for s in ['left','right']]
            sensorfeet=[sn.index(f'{s}_ankle_roll_link') for s in ['left','right']]
            f=z['net_contact_forces'][:,:,sensorfeet]
            force=np.linalg.norm(f,axis=-1);contact=force>10
            rootdelta=delta[:,:,bn.index('pelvis')]
            referenceq=np.c_[ref['body_pos_w'][:,0],ref['body_quat_w'][:,0],ref['joint_pos']]
            reftf=fk(referenceq,jn);com,mass=center_of_mass(reftf)
            refmin=np.stack([sole_minimum(reftf,s) for s in ['left','right']],axis=-1)
            actualq=np.c_[z['root_state'][:,0,:3]-z['env_origins'][0],z['root_state'][:,0,3:7],z['joint_pos'][:,0]]
            acttf=fk(actualq,jn);actmin=np.stack([sole_minimum(acttf,s) for s in ['left','right']],axis=-1)
            valid0=valid[:,0];n=int(valid0.sum())
            err=z['errors'][:,0]
            strict_failed=(err[:,0]>=.3)|(err[:,2]>=.15)|(err[:,3]>=.15)|(actual[:,0,0,2]<.35)|z['termination_terms'][:,0].any(-1)
            strict_t=first(strict_failed&valid0,fps)
            # The legacy first-failure array uses the horizon as a sentinel on success.
            # Do not report that sentinel as an actual failure time.
            frames=sorted(set([1,10,25,n]+([int(round(strict_t*fps))] if strict_t is not None else [])))
            snapshots=[]
            for step in frames:
                ix=step-1
                if ix<0 or ix>=n:continue
                top=np.argsort(joint_abs[ix,0])[-5:][::-1]
                snapshots.append(dict(time_s=step/fps,root_xyz_error_m=rootdelta[ix,0].tolist(),
                    root_error_m=float(np.linalg.norm(rootdelta[ix,0])),
                    joint_mae_rad=float(joint_abs[ix,0].mean()),
                    top_joint_errors=[dict(joint=jn[j],error_rad=float(joint_abs[ix,0,j])) for j in top],
                    foot_force_xyz_N=f[ix,0].tolist(),
                    reference_foot_capsule_min_z_m=refmin[step].tolist(),
                    actual_foot_capsule_min_z_m=actmin[ix].tolist(),
                    foot_position_error_m=np.linalg.norm(delta[ix,0,feet],axis=-1).tolist(),
                    termination_terms=[tn[i] for i,v in enumerate(z['termination_terms'][ix,0]) if v]))
            limits=z['effort_limits'];vel_limits=z['velocity_limits']
            validjoint=np.broadcast_to(valid[:,:,None],joint_abs.shape)
            sat=(np.abs(z['applied_torque'])>=.95*limits[None])&validjoint
            term_counts={term:int((z['termination_terms'][:,:,i]&valid).sum()) for i,term in enumerate(tn)}
            support_stats=[]
            for side,index in zip(['left','right'],range(2)):
                support_stats.append(dict(side=side,contact_fraction=float(contact[:,:,index][valid].mean()),
                    mean_force_N=float(force[:,:,index][valid].mean()),
                    pre_strict_failure_or_end_contact_fraction=float(contact[:max(1,int(strict_t*fps)) if strict_t is not None else n,0,index].mean())))
            row=dict(name=job['name'],checkpoint_sha256=job['checkpoint_sha256'],reference_sha256=job['reference_sha256'],trace_sha256=metrics['trace_sha256'],
                telemetry_reproduction={key:dict(old=old[key],new=metrics[key]) for key in ['completion_fraction','strict_every_frame_pass_fraction','mean_active_errors']},
                first_strict_failure_env0_s=strict_t,env0_valid_duration_s=n/fps,
                first_root_over15cm_env0_s=first((np.linalg.norm(rootdelta[:,0],axis=-1)>=.15)&valid0),
                first_joint_mae_over03_env0_s=first((joint_abs[:,0].mean(-1)>=.3)&valid0),
                initial_root_error_m=float(np.linalg.norm(z['initial_root_state'][0,:3]-z['env_origins'][0]-ref['body_pos_w'][0,0])),
                initial_joint_max_error_rad=float(np.abs(z['initial_joint_pos'][0]-ref['joint_pos'][0]).max()),
                initial_reference_foot_min_z_m=refmin[0].tolist(),
                reference_min_foot_z_range_m=[float(refmin.min()),float(np.min(refmin,axis=1).max())],
                initial_reference_com_world=com[0].tolist(),mass_kg=mass,
                torque_near_limit_fraction=float(sat.sum()/validjoint.sum()),
                velocity_exceedance_fraction=float(((np.abs(z['joint_vel'])>vel_limits[None])&validjoint).sum()/validjoint.sum()),
                termination_counts=term_counts,support=support_stats,snapshots=snapshots)
            rows.append(row)
    result=dict(scope='Four development motions, fixed final policies, nominal condition, no training. Force observations diagnose execution, not reference truth.',
        geometry_caveat='URDF cylinder-to-capsule lower-envelope proxy; not ground-truth foot-ground contact. No correction from policy output.',
        telemetry_caveat='Contact sensor reports aggregate normal contact forces, not isolated foot-ground friction. Implicit-actuator applied_torque is an estimated telemetry quantity, not independently measured joint torque.',
        supersedes='support_analysis.json: successful horizon sentinel was incorrectly printed as first strict failure; v2 uses per-frame predicates and null for no failure. Evaluation metrics unchanged.',
        script_sha256=sha(Path(__file__)),rows=rows)
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
