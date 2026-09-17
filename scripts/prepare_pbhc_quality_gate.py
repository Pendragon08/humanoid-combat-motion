"""Source-only temporal projection and conservative whole-trajectory acceptance.

The previous geometric candidate supplies a direction, not an accepted target.
No rollout/checkpoint/evaluation score is read. All failed sources fall back to raw.
"""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import osqp
from scipy import sparse
from audit_pbhc_reference_kinematics import tree
from audit_pbhc_temporal_quality import geometry,derivative

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_quality_gate_20260917'
OUT=ROOT/'runs/pbhc_quality_gate_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def matrices(n):
    return [sparse.csc_matrix(np.diff(np.eye(n),n=k,axis=0)) for k in range(4)]

def qp(target,lower,upper,ds,speed,acceleration,jerk,source=None,cfg=None):
    d0,d1,d2,d3=ds
    P=2*(d0+20*d1.T@d1+500*d2.T@d2)
    A=sparse.vstack(ds,format='csc')
    l=np.r_[lower,np.full(d1.shape[0],-speed/50),np.full(d2.shape[0],-acceleration/2500),np.full(d3.shape[0],-jerk/125000)]
    u=np.r_[upper,-l[len(lower):]]
    if source is not None:
        vmax=np.max(np.abs(d1@source))*cfg['joint_peak_relative_tolerance']+cfg['joint_velocity_absolute_tolerance_rad_s']/50
        amax=np.max(np.abs(d2@source))*cfg['joint_peak_relative_tolerance']+cfg['joint_acceleration_absolute_tolerance_rad_s2']/2500
        A=sparse.vstack([A,d1,d2],format='csc')
        l=np.r_[l,-vmax-d1@source,-amax-d2@source];u=np.r_[u,vmax-d1@source,amax-d2@source]
    solver=osqp.OSQP();solver.setup(P=sparse.triu(P,format='csc'),q=-2*target,A=A,l=l,u=u,
        verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=100000,polishing=True)
    r=solver.solve();status=r.info.status
    if r.x is None or status not in ['solved','solved inaccurate']:
        return None,dict(status=status,iterations=r.info.iter)
    violation=float(max(np.maximum(l-A@r.x,0).max(),np.maximum(A@r.x-u,0).max()))
    if violation>2e-8:return None,dict(status=status,iterations=r.info.iter,constraint_violation=violation)
    return r.x,dict(status=status,iterations=r.info.iter,constraint_violation=violation)

def check(raw,candidate,names,contacts,onset,cfg):
    feet0,low0=geometry(raw,names);feet,low=geometry(candidate,names)
    penetration0=np.maximum(-low0,0);penetration=np.maximum(-low,0)
    delta=candidate[:,7:]-raw[:,7:];dz=candidate[:,2]-raw[:,2]
    known=np.isclose(contacts,1);active=np.arange(len(raw))/50>=onset
    upper=[i for i,n in enumerate(names) if not any(k in n for k in ['hip_','knee_','ankle_'])]
    v0=np.abs(derivative(raw[:,7:],1)).max(0);a0=np.abs(derivative(raw[:,7:],2)).max(0)
    v=np.abs(derivative(candidate[:,7:],1)).max(0);a=np.abs(derivative(candidate[:,7:],2)).max(0)
    stats=dict(joint_delta_rad=float(np.abs(delta).max()),
        joint_delta_velocity_rad_s=float(np.abs(derivative(delta,1)).max()),
        joint_delta_acceleration_rad_s2=float(np.abs(derivative(delta,2)).max()),
        joint_delta_jerk_rad_s3=float(np.abs(derivative(delta,3)).max()),
        root_lift_m=float(dz.max()),root_min_lift_m=float(dz.min()),
        root_delta_velocity_m_s=float(np.abs(derivative(dz,1)).max()),
        root_delta_acceleration_m_s2=float(np.abs(derivative(dz,2)).max()),
        root_delta_jerk_m_s3=float(np.abs(derivative(dz,3)).max()),
        foot_xy_change_m=float(np.linalg.norm(feet[:,:,:2]-feet0[:,:,:2],axis=-1).max()),
        original_penetration_max_m=float(penetration0.max()),candidate_penetration_max_m=float(penetration.max()),
        original_penetration_mean_m=float(penetration0.mean()),candidate_penetration_mean_m=float(penetration.mean()),
        worst_penetration_regression_m=float(np.maximum(penetration-penetration0,0).max()),
        known_contact_gap_regression_m=float(np.abs(low[known]).mean()-np.abs(low0[known]).mean()) if known.any() else None)
    checks={key:stats[key]<=cfg[key]+cfg['numerical_tolerance'] for key in [
        'joint_delta_rad','joint_delta_velocity_rad_s','joint_delta_acceleration_rad_s2','joint_delta_jerk_rad_s3',
        'root_lift_m','root_delta_velocity_m_s','root_delta_acceleration_m_s2','root_delta_jerk_m_s3','foot_xy_change_m']}
    checks.update(root_no_downshift=stats['root_min_lift_m']>=-1e-7,
        root_xy_orientation_unchanged=bool(np.array_equal(candidate[:,:2],raw[:,:2]) and np.array_equal(candidate[:,3:7],raw[:,3:7])),
        initial_final_three_frames_unchanged=bool(np.array_equal(candidate[:3],raw[:3]) and np.array_equal(candidate[-3:],raw[-3:])),
        active_and_recovery_joints_unchanged=bool(np.array_equal(candidate[active,7:],raw[active,7:])),
        upper_joints_unchanged=bool(np.array_equal(candidate[:,7+np.array(upper)],raw[:,7+np.array(upper)])),
        per_joint_peak_velocity=bool(np.all(v<=v0*cfg['joint_peak_relative_tolerance']+cfg['joint_velocity_absolute_tolerance_rad_s']+cfg['numerical_tolerance'])),
        per_joint_peak_acceleration=bool(np.all(a<=a0*cfg['joint_peak_relative_tolerance']+cfg['joint_acceleration_absolute_tolerance_rad_s2']+cfg['numerical_tolerance'])),
        pointwise_no_worse_penetration=stats['worst_penetration_regression_m']<=cfg['penetration_regression_tolerance_m'],
        mean_penetration_reduction=stats['candidate_penetration_mean_m']<=stats['original_penetration_mean_m']*(1-cfg['minimum_full_mean_penetration_relative_reduction']),
        known_contact_gap=(stats['known_contact_gap_regression_m'] is not None and stats['known_contact_gap_regression_m']<=cfg['known_contact_gap_regression_tolerance_m']))
    return dict(pass_gate=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},metrics=stats)

def main():
    cfgpath=ROOT/'configs/pbhc_quality_gate_v1.json';cfg=json.loads(cfgpath.read_text())
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True);REC.mkdir(parents=True,exist_ok=True)
    # Freeze the prospective protocol before reading any new candidate execution.
    (REC/'frozen_config.json').write_text(json.dumps(cfg,indent=2))
    prior=json.loads((ROOT/'records/pbhc_support_repair_20260917/protocol.json').read_text())
    cases=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())['motions']
    outputs=[];retrospective=[]
    for entry in [r for r in prior['records'] if r['arm']=='prep_support']:
        name=entry['name'];source=Path(entry['source'])
        assert sha(source)==entry['source_sha256']
        with np.load(source,allow_pickle=False) as z:
            names=list(z['joint_names']);raw=np.c_[z['body_pos_w'][:,0],z['body_quat_w'][:,0],z['joint_pos']].astype(float)
        with np.load(entry['input'],allow_pickle=False) as z:old=z['qpos'].astype(float)
        assert sha(Path(entry['input']))==entry['input_sha256']
        case=next(c for c in cases if c['name']==name)
        contact=np.asarray(next(iter(joblib.load(case['source_path']).values()))['contact_mask'])
        ids=np.minimum(np.rint(np.arange(len(raw))/50*case['source_fps']).astype(int),len(contact)-1);contact=contact[ids]
        retrospective.append(dict(name=name,arm='previous_prep_support',**check(raw,old,names,contact,entry['onset_s'],cfg)))
        n=len(raw);ds=matrices(n);candidate=raw.copy();solver_records=[]
        frozen=(np.arange(n)/50>=entry['onset_s']);frozen[:3]=True;frozen[-3:]=True
        failed=False
        for j,joint in enumerate(names):
            if not any(s in joint for s in ['hip_','knee_','ankle_']):continue
            limit=tree.find(f"joint[@name='{joint}']/limit")
            lo=np.maximum(-cfg['joint_delta_rad'],float(limit.get('lower'))-raw[:,7+j]);hi=np.minimum(cfg['joint_delta_rad'],float(limit.get('upper'))-raw[:,7+j])
            lo[frozen]=0;hi[frozen]=0
            x,record=qp(old[:,7+j]-raw[:,7+j],lo,hi,ds,cfg['joint_delta_velocity_rad_s'],cfg['joint_delta_acceleration_rad_s2'],cfg['joint_delta_jerk_rad_s3'],raw[:,7+j],cfg)
            solver_records.append(dict(joint=joint,**record))
            if x is None:failed=True;break
            x[frozen]=0;candidate[:,7+j]+=x
        if not failed:
            _,oldlow=geometry(raw,names);_,newlow=geometry(candidate,names)
            required=np.maximum(0,np.max(np.minimum(oldlow,0)-newlow,axis=1));required[:3]=0;required[-3:]=0
            upper=np.full(n,cfg['root_lift_m']);upper[:3]=0;upper[-3:]=0
            x,record=qp(np.zeros(n),required,upper,ds,cfg['root_delta_velocity_m_s'],cfg['root_delta_acceleration_m_s2'],cfg['root_delta_jerk_m_s3'])
            solver_records.append(dict(joint='root_z_source_only_compensation',**record))
            if x is None:failed=True
            else:
                x[:3]=0;x[-3:]=0;x=np.maximum(x,0);candidate[:,2]+=x
        gate=check(raw,candidate,names,contact,entry['onset_s'],cfg)
        if failed:gate['pass_gate']=False;gate['checks']['optimizer_completed']=False
        for kind,q in [('candidate',candidate),('selected',candidate if gate['pass_gate'] else raw)]:
            folder=OUT/kind;folder.mkdir(exist_ok=True)
            path=folder/f'sample_{name}.npz';np.savez_compressed(path,qpos=q,fps=50.,joint_names=np.array(names))
        outputs.append(dict(name=name,source=str(source),source_sha256=sha(source),previous_candidate_sha256=entry['input_sha256'],
            source_contact_path=case['source_path'],source_contact_sha256=sha(Path(case['source_path'])),onset_s=entry['onset_s'],
            selected='projected_candidate' if gate['pass_gate'] else 'raw_fallback',
            candidate=str(OUT/'candidate'/f'sample_{name}.npz'),selected_input=str(OUT/'selected'/f'sample_{name}.npz'),
            candidate_sha256=sha(OUT/'candidate'/f'sample_{name}.npz'),selected_sha256=sha(OUT/'selected'/f'sample_{name}.npz'),
            gate=gate,solvers=solver_records))
        print(name,outputs[-1]['selected'],[k for k,v in gate['checks'].items() if not v],gate['metrics'],flush=True)
    manifest=dict(scope='Source-only development gate. No execution score read; original fallback retained for every rejected motion.',
        config_sha256=sha(cfgpath),script_sha256=sha(Path(__file__)),records=outputs,retrospective_prior_gate=retrospective,
        gate_pass_does_not_prove=['Correct contact labels','Dynamics feasibility','Tracking success','Training utility','Generalization'])
    (REC/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
