"""Preserve strict scores and add common23DOF / local skill diagnostics."""
import json,hashlib,zipfile,csv
from pathlib import Path
import numpy as np
import mujoco
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from combat.twist_tracking import ROOT,scene
from scripts.pbhc_refinement_pilot import WINDOWS

def main():
    out=ROOT/'records/tracker_upgrade_20260916'
    names=list(WINDOWS)
    old_model,_=scene(1)
    twist2=mujoco.MjModel.from_xml_path(str(ROOT/'third_party/TWIST2/assets/g1/g1_sim2sim_29dof.xml'))
    sonic=mujoco.MjModel.from_xml_path(str(ROOT/'third_party/HumanTracker/storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml'))
    rows=[]
    for label,folder,model,prefix in [('TWIST25','pbhc_seed_twist_20260916',old_model,'a0_'),
        ('TWIST2_20k100','twist2_upgrade_20260916',twist2,''),('TWIST2_25k50','twist2_upgrade_20260916',twist2,''),
        ('TWIST2_20k50','twist2_upgrade_20260916',twist2,''),('TWIST2_25k100','twist2_upgrade_20260916',twist2,''),
        ('SONIC','sonic_probe_20260916',sonic,'')]:
        for name in names:
            suffix={'TWIST2_20k100':'_20k_100','TWIST2_25k50':'_25k_50','TWIST2_20k50':'_20k_50','TWIST2_25k100':'_25k_100'}.get(label,'')
            path=ROOT/'runs'/folder/(name+suffix);m=json.loads((path/'metrics.json').read_text())
            with np.load(path/'rollout.npz') as z:
                q=z['qpos'].copy();r=z['reference'].copy();je=z['joint_error'].reshape(-1);re=z['root_error'].reshape(-1);be=z['body_error'].reshape(-1);falls=z['falls'].reshape(-1);strict=z['strict'].reshape(-1)
            if q.ndim==3:q=q[:,0];r=r[:,0]
            joint_names=m['joint_names'];indices=[j for j,n in enumerate(joint_names) if 'wrist' not in n]
            assert len(indices)==23
            common=np.abs(q[:,7:][:,indices]-r[:,7:][:,indices]).mean(1)
            effectors=['left_wrist_roll_link','right_wrist_roll_link','left_ankle_roll_link','right_ankle_roll_link'];ids=[model.body(prefix+n).id for n in effectors];data=mujoco.MjData(model)
            def fk(a):
                points=[]
                for row in a:data.qpos[:]=row;mujoco.mj_forward(model,data);points.append(data.xpos[ids].copy())
                rot=Rotation.from_quat(a[:,[4,5,6,3]]).as_matrix()
                return np.einsum('tji,tkj->tki',rot,np.asarray(points)-a[:,:3,None].transpose(0,2,1))
            pp,rr=fk(q),fk(r);times=(np.arange(len(q))+1)*.02;lo,hi=WINDOWS[name];active=(times>=lo)&(times<=hi)
            rv=np.linalg.norm(savgol_filter(rr,11,3,deriv=1,delta=.02,axis=0),axis=-1);pv=np.linalg.norm(savgol_filter(pp,11,3,deriv=1,delta=.02,axis=0),axis=-1)
            eff=max([0,1] if 'punch' in name else [2,3],key=lambda e:np.quantile(rv[active,e],.95));start=np.flatnonzero(active)[0]
            excursion=lambda x:float(np.linalg.norm(x[active,eff]-x[start,eff],axis=-1).max())
            yawq=Rotation.from_quat(q[:,[4,5,6,3]]).as_euler('xyz')[:,2];yawr=Rotation.from_quat(r[:,[4,5,6,3]]).as_euler('xyz')[:,2]
            first=lambda seq:float((np.flatnonzero(seq)[0]+1)*.02) if np.any(seq) else None
            rows.append({'tracker':label,'name':name,'no_fall':bool(~falls.any()),'strict':bool(strict.all()),'strict_common23':bool(((common<.3)&(re<.15)&(be<.15)&~falls).all()),
                'strict_prefix_s':float(np.logical_and.accumulate(strict).sum()*.02),'joint_mae_rad':float(je.mean()),'common23_mae_rad':float(common.mean()),
                'root_error_m':float(re.mean()),'root_xy_error_m':float(np.linalg.norm(q[:,:2]-r[:,:2],axis=1).mean()),'root_z_error_m':float(abs(q[:,2]-r[:,2]).mean()),
                'yaw_error_rad':float(abs(np.arctan2(np.sin(yawq-yawr),np.cos(yawq-yawr))).mean()),'body_error_m':float(be.mean()),
                'first_joint_failure_s':first(je>=.3),'first_root_failure_s':first(re>=.15),'first_body_failure_s':first(be>=.15),'first_fall_s':first(falls),
                'local_effector_error_m':float(np.linalg.norm(pp[active,eff]-rr[active,eff],axis=-1).mean()),'speed_ratio':float(np.quantile(pv[active,eff],.95)/np.quantile(rv[active,eff],.95)),
                'excursion_ratio':excursion(pp)/excursion(rr),'selected_effector':effectors[eff],'torque_clip_substep_fraction':m.get('torque_clip_substep_fraction'),
                'result_path':str(path),'result_sha256':hashlib.sha256((path/'rollout.npz').read_bytes()).hexdigest()})
    agg=[]
    for label in dict.fromkeys(x['tracker'] for x in rows):
        group=[x for x in rows if x['tracker']==label]
        agg.append({'tracker':label,'n':len(group),'no_fall':sum(x['no_fall'] for x in group),'strict':sum(x['strict'] for x in group),'strict_common23':sum(x['strict_common23'] for x in group),
                    **{k:float(np.mean([x[k] for x in group])) for k in ['strict_prefix_s','joint_mae_rad','common23_mae_rad','root_error_m','root_xy_error_m','root_z_error_m','yaw_error_rad','body_error_m','local_effector_error_m','speed_ratio','excursion_ratio']}})
    result={'scope':'Development controller-system diagnostic, differing native models/reset/gains; no isolated algorithm ranking or independent generalization claim',
        'posthoc':'Common23DOF and local skill metrics descriptive, prior strict scores preserved;2x2 TWIST2 frequency/checkpoint crossover added after initial2 configurations',
        'rows':rows,'aggregate':agg}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(agg,indent=2))
    with zipfile.ZipFile(out/'results_archive.zip','w',zipfile.ZIP_DEFLATED) as z:
        for run in ['twist2_upgrade_20260916','sonic_probe_20260916']:
            for f in (ROOT/'runs'/run).rglob('*'):
                if f.is_file() and f.suffix in ['.json','.npz','.py']:z.write(f,str(f.relative_to(ROOT/'runs')))
        z.write(out/'summary.json','summary.json');z.write(out/'summary.csv','summary.csv')
if __name__=='__main__':main()
