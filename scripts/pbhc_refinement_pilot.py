"""Development-only bounded root-height correction, not a full motion generator.

Use the same four full combat candidates, frozen controller and metrics. Compare
unmodified, PBHC-inspired grounding, and matched optimization +/- phase fidelity.
The unknown0.5 labels in upstream transition padding are not treated as contacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import joblib
import mujoco
import numpy as np
from scipy.optimize import minimize
from combat.twist_tracking import ROOT,scene

PANEL=ROOT/'runs/pbhc_seed_panel_20260916'
OUT=ROOT/'runs/pbhc_refinement_pilot_20260916'
# Coarse development windows inspected from16-frame sheets; not expert truth.
WINDOWS={'Horse-stance_punch':[1.3,4.4],'Hooks_punch':[.77,4.63],
         'Side_kick':[1.57,3.13],'Roundhouse_kick':[1.37,2.77]}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True);(OUT/'references').mkdir();(OUT/'logs').mkdir()
    model,_=scene(1);data=mujoco.MjData(model)
    feet=[[g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name==f'a0_{side}_ankle_roll_link'
           and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE] for side in ['left','right']]
    assert [len(x) for x in feet]==[4,4]
    inventory=json.loads((PANEL/'inventory.json').read_text())
    cases=[];diagnostics=[]
    for item in inventory['motions']:
        name=item['name']
        if name not in WINDOWS:continue
        path=Path(item['target_reference'])
        with np.load(path,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
        assert sha(path)==item['target_sha256']
        pelvis=list(base['body_names']).index('pelvis')
        q=np.c_[base['body_pos_w'][:,pelvis],base['body_quat_w'][:,pelvis],base['joint_pos']]
        t=np.arange(len(q))/50
        raw=next(iter(joblib.load(item['source_path']).values()))
        ix=np.minimum(np.rint(t*raw['fps']).astype(int),len(raw['contact_mask'])-1)
        contact=np.asarray(raw['contact_mask'])[ix]
        bottom=[]
        for row in q:
            data.qpos[:]=row;mujoco.mj_forward(model,data)
            bottom.append([min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in gs) for gs in feet])
        bottom=np.array(bottom)
        known=np.isclose(contact,1.).astype(float)
        weights=known.sum(1)
        target=np.divide(-(bottom*known).sum(1),weights,out=np.zeros(len(q)),where=weights>0)
        active=(t>=WINDOWS[name][0])&(t<=WINDOWS[name][1])
        variants={'raw':np.zeros(len(q))}
        any_contact=(contact!=0).any(1)
        offsets=np.zeros(len(q))
        for k in range(len(q)):
            offsets[k]=bottom[k].min() if any_contact[k] else offsets[k-1] if k else 0.
        height=q[:,2]-offsets
        for k in range(1,len(q)):height[k]=.3*height[k]+.7*height[k-1]
        variants['grounding_ema']=height-q[:,2]
        solver={}
        for protected in [False,True]:
            # Same candidate, variables, bounds and iteration limit. Change
            # only phase-dependent penalties on correction and its derivative.
            fidelity=2.+(20.*active if protected else 0.)
            smooth=50.+(300.*(active[:-1]|active[1:]) if protected else np.zeros(len(q)-1))
            def objective(d):
                diff=np.diff(d);res=d-target
                value=np.sum(weights*res**2)+np.sum(fidelity*d*d)+np.sum(smooth*diff*diff)
                grad=2*weights*res+2*fidelity*d
                grad[:-1]-=2*smooth*diff;grad[1:]+=2*smooth*diff
                return value,grad
            fit=minimize(objective,np.zeros(len(q)),jac=True,method='L-BFGS-B',bounds=[(-.04,.04)]*len(q),
                         options={'maxiter':200,'ftol':1e-12,'gtol':1e-8})
            arm='phase_fidelity' if protected else 'bounded_plain'
            variants[arm]=fit.x
            solver[arm]={'success':bool(fit.success),'iterations':int(fit.nit),'message':str(fit.message),'objective':float(fit.fun)}
            if not fit.success:raise RuntimeError((name,arm,fit.message))
        for arm,delta in variants.items():
            ref={k:v.copy() for k,v in base.items()}
            ref['body_pos_w'][:,:,2]+=delta[:,None]
            ref['body_lin_vel_w'][:,:,2]+=np.gradient(delta,.02)[:,None]
            np.testing.assert_array_equal(ref['joint_pos'],base['joint_pos'])
            np.testing.assert_array_equal(ref['body_quat_w'],base['body_quat_w'])
            targetpath=OUT/'references'/f'{name}_{arm}.npz'
            np.savez_compressed(targetpath,**ref)
            cases.append({'id':f'{name}_{arm}','name':name,'arm':arm,'path':str(targetpath),'sha256':sha(targetpath)})
            corrected=bottom+delta[:,None]
            known_mask=known.astype(bool)
            diagnostics.append({'name':name,'arm':arm,'source_sha256':item['source_sha256'],
                'duration_s':float(t[-1]),'active_window_s':WINDOWS[name],
                'unknown_contact_fraction':float(np.isclose(contact,.5).mean()),
                'maximum_height_change_m':float(abs(delta).max()),
                'active_max_vertical_speed_change_m_s':float(abs(np.gradient(delta,.02)[active]).max()),
                'known_support_height_mae_m':float(abs(corrected[known_mask]).mean()) if known_mask.any() else None,
                'maximum_foot_penetration_m':float(max(0,-corrected.min())),
                'joint_angles_identical':True,'duration_identical':True,'solver':solver.get(arm)})
    write(OUT/'protocol.json',{'stage':'S2-light development component pilot; not full S1--S4',
        'source_windows':WINDOWS,'window_provenance':'Assistant inspected upstream-labelled contact sheets; coarse, not independently annotated truth',
        'arms':['raw','grounding_ema','bounded_plain','phase_fidelity'],
        'grounding_ema':'PBHC correct_motion/EMA idea, alpha0.3, approximated with robot foot collision spheres instead of human mesh; not original pipeline reproduction',
        'optimization':'one root-height correction per frame, +/-4cm; known contact==1, unknown0.5 ignored; data fidelity2 and smoothness50; protected adds20 and300 inside active windows; 200iteration cap',
        'hypothesis':'Phase-dependent penalties reduce distortion; execution gain not assumed',
        'controller':'Same frozen TWIST25DOF policy, no training, seed123',
        'evaluation':'Old strict global score retained; report fall, joint errors and end-effector fidelity separately',
        'data_partition':'four development examples; no test-set or generalization claim',
        'only_modified_quantity':'root z and derivative; full joint-space/contact-force optimization absent',
        'script_sha256':sha(Path(__file__)),'cases':cases})
    write(OUT/'reference_quality.json',diagnostics)
    print('REFINEMENT_PILOT_READY',len(cases),flush=True)

def run(shard):
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['script_sha256']==sha(Path(__file__))
    states=[]
    for i,c in enumerate(protocol['cases']):
        if i%2!=shard:continue
        p=Path(c['path']);assert sha(p)==c['sha256']
        output=OUT/'results'/c['id']
        if output.exists():raise FileExistsError(output)
        start=time.time()
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_twist.py'),'--reference',str(p),
             '--device',f'cuda:{shard}','--output',str(output)]
        with (OUT/'logs'/(c['id']+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        states.append({'id':c['id'],'command':cmd,'exit':result.returncode,'elapsed_s':time.time()-start})
        write(OUT/f'queue_{shard}.json',states)
        if result.returncode:raise RuntimeError(c['id'])
        print('REFINEMENT_CASE_COMPLETE',c['id'],flush=True)
    write(OUT/f'COMPLETE_{shard}.json',{'completed':len(states)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--shard',type=int,default=0,choices=[0,1]);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:run(a.shard)
