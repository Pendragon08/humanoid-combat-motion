"""Source-only preparation repair pilot; never uses policy rollout as a target."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from audit_pbhc_reference_kinematics import fk,tree
from analyze_pbhc_support_diagnostics import sole_minimum

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/pbhc_support_repair_20260917'
REC=ROOT/'records/pbhc_support_repair_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if OUT.exists() or REC.exists():raise FileExistsError('Preserve existing pilot')
    OUT.mkdir(parents=True);REC.mkdir(parents=True);dest=OUT/'named_qpos';dest.mkdir()
    inventory=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())
    windows=json.loads((ROOT/'runs/pbhc_refinement_pilot_20260916/protocol.json').read_text())['source_windows']
    jobs=[]
    for case in inventory['motions']:
        if case['category']!='combat_candidate':continue
        name=case['name'];source=ROOT/'runs/pbhc_beyondmimic_20260916/references_full'/f'sample_{name}_actor0.npz'
        with np.load(source,allow_pickle=False) as z:
            names=list(z['joint_names']);q=np.c_[z['body_pos_w'][:,0],z['body_quat_w'][:,0],z['joint_pos']].astype(float)
        orig=q.copy();tf=fk(q,names)
        initial_low=min(sole_minimum(tf,s)[0] for s in ['left','right'])
        assert abs(initial_low)<.025,initial_low
        aligned=q.copy();aligned[:,2]-=initial_low
        joints=[i for i,n in enumerate(names) if any(x in n for x in ['hip_','knee_','ankle_'])];assert len(joints)==12
        onset=windows[name][0];ids=np.flatnonzero((np.arange(len(q))/50>0)&(np.arange(len(q))/50<onset))
        lower=[];upper=[]
        for j in joints:
            lim=tree.find(f"joint[@name='{names[j]}']/limit");lower.append(float(lim.get('lower')));upper.append(float(lim.get('upper')))
        contacts=np.asarray(next(iter(joblib.load(case['source_path']).values()))['contact_mask'])
        ci=np.minimum(np.rint(ids/50*case['source_fps']).astype(int),len(contacts)-1)
        known=np.isclose(contacts[ci],1)
        base=aligned[ids].copy();base_tf=fk(base,names)
        foot_names=['left_ankle_roll_link','right_ankle_roll_link']
        goals=np.stack([base_tf[n][:,:3,3] for n in foot_names],axis=1)
        reference_rot=np.stack([base_tf[n][:,:3,:3] for n in foot_names],axis=1)
        floor=np.stack([sole_minimum(base_tf,s) for s in ['left','right']],axis=1)
        dz=np.maximum(-floor,0)
        stance=known&(floor<.025)
        dz[stance]=-floor[stance]
        goals[:,:,2]+=dz
        # Hard 0.12rad cap tapering to zero at both ends of the preparation window.
        t=ids/50;taper=np.minimum(np.minimum(t/.10,(onset-t)/.20),1).clip(0,1)
        cap=.12*taper[:,None]
        lo=np.maximum(-cap,np.asarray(lower)[None]-base[:,7+np.array(joints)])
        hi=np.minimum(cap,np.asarray(upper)[None]-base[:,7+np.array(joints)])
        assert np.all(lo<hi)
        n=len(ids);d=12
        def residual(x):
            delta=x.reshape(n,d);p=base.copy();p[:,7+np.array(joints)]+=delta;tf=fk(p,names)
            pos=np.stack([tf[k][:,:3,3] for k in foot_names],axis=1)
            rot=np.stack([tf[k][:,:3,:3] for k in foot_names],axis=1)
            angle=Rotation.from_matrix((reference_rot.swapaxes(-2,-1)@rot).reshape(-1,3,3)).as_rotvec().reshape(n,6)
            local=np.c_[((pos-goals)*np.array([120.,120.,100.])).reshape(n,6),angle*8,delta*2]
            smooth=np.diff(np.vstack([np.zeros((1,d)),delta,np.zeros((1,d))]),axis=0)*12
            return np.r_[local.ravel(),smooth.ravel()]
        sparsity=lil_matrix((n*24+(n+1)*12,n*12),dtype=int)
        for i in range(n):sparsity[i*24:(i+1)*24,i*12:(i+1)*12]=1
        for i in range(n+1):
            for j in [i-1,i]:
                if 0<=j<n:sparsity[n*24+i*12:n*24+(i+1)*12,j*12:(j+1)*12]=np.eye(12)
        fit=least_squares(residual,np.zeros(n*d),bounds=(lo.ravel(),hi.ravel()),jac_sparsity=sparsity.tocsr(),max_nfev=40,
                          ftol=1e-6,xtol=1e-6,gtol=1e-6)
        repaired=aligned.copy();repaired[np.ix_(ids,7+np.array(joints))]+=fit.x.reshape(n,d)
        assert np.array_equal(repaired[np.arange(len(q))/50>=onset,7:],orig[np.arange(len(q))/50>=onset,7:])
        assert np.max(np.abs(repaired[:,7:]-orig[:,7:]))<=.120001
        for arm,p in [('ground_align',aligned),('prep_support',repaired)]:
            out=dest/f'sample_{name}_{arm}.npz';np.savez_compressed(out,qpos=p,fps=50.,joint_names=np.array(names))
            tf=fk(p,names);oldtf=fk(orig,names)
            feet=np.stack([tf[k][:,:3,3] for k in foot_names],axis=1);oldfeet=np.stack([oldtf[k][:,:3,3] for k in foot_names],axis=1)
            floor_new=np.stack([sole_minimum(tf,s) for s in ['left','right']],axis=1)
            jobs.append(dict(name=name,arm=arm,source=str(source),source_sha256=sha(source),input=str(out),input_sha256=sha(out),frames=len(q),
                initial_ground_shift_m=-float(initial_low),max_joint_change_rad=float(np.max(np.abs(p[:,7:]-orig[:,7:]))),
                max_foot_xy_change_m=float(np.linalg.norm(feet[:,:,:2]-oldfeet[:,:,:2],axis=-1).max()),
                max_foot_xyz_change_m=float(np.linalg.norm(feet-oldfeet,axis=-1).max()),
                preparation_foot_penetration_before_m=float(np.maximum(-floor,0).max()),
                preparation_foot_penetration_after_m=float(np.maximum(-floor_new[ids],0).max()),
                active_joint_trajectory_unchanged=True,onset_s=onset,optimizer_success=bool(fit.success),optimizer_nfev=fit.nfev,
                optimizer_message=fit.message))
        print('PREPARATION_REPAIR_CREATED',name,fit.nfev,fit.message,flush=True)
    protocol=dict(scope='Development-only source-geometry repair ablation; fixed checkpoints; not data-utility or novelty evidence',
        arms=['raw_existing','ground_align','prep_support'],records=jobs,script_sha256=sha(Path(__file__)),
        invariants=['No rollout used as target','Same full duration/frame rate/root XY and root orientation',
                    'Upper-body joint trajectories unchanged','Active skill and recovery joint trajectories unchanged',
                    'All four sources retained','No unknown contact relabeling','Strict evaluation thresholds unchanged'],
        limitations=['URDF capsule-envelope geometry proxy, not ground truth contact',
                     'Small source-only repair may not resolve horizontal drift',
                     'Frozen per-motion policy may not generalize to modified input; negative result does not prove untrackability',
                     'Preparation window is coarse development annotation',
                     'Report evaluation against BOTH repaired and original reference'])
    (REC/'protocol.json').write_text(json.dumps(protocol,indent=2))

if __name__=='__main__':main()
