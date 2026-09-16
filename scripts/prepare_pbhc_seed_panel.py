"""Audit bundled PBHC demonstrations and render complete-motion contact sheets.

Labels come from upstream names. Source identity independence and semantic
boundaries remain separate review fields; no aliases become extra sources.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image, ImageDraw
from combat.twist_tracking import ROOT, scene, Controller
from scripts.single_robot_s0 import arrays

UP=ROOT/'third_party/PBHC'
OUT=ROOT/'runs/pbhc_seed_panel_20260916'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True);(OUT/'references').mkdir();(OUT/'review').mkdir()
    model,_=scene(1);controller=Controller(model,0,None,'cpu')
    xml=UP/'description/robots/g1/g1_23dof_lock_wrist.xml'
    native=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(native)
    names=[native.joint(i).name for i in range(native.njnt) if native.jnt_type[i]!=0]
    assert len(names)==23
    bodyids=[i for i in range(model.nbody) if model.body(i).name.startswith('a0_')]
    renderer=mujoco.Renderer(native,height=240,width=320)
    camera=mujoco.MjvCamera();camera.azimuth=130;camera.elevation=-15;camera.distance=2.8
    rows=[]
    combat={'Hooks_punch','Horse-stance_punch','Side_kick','Roundhouse_kick'}
    for path in sorted((UP/'example/motion_data').glob('*.pkl')):
        raw=joblib.load(path)
        assert len(raw)==1
        source_key,values=next(iter(raw.items()))
        j=np.asarray(values['dof']);pos=np.asarray(values['root_trans_offset']);aa=np.asarray(values['pose_aa'])
        fps=float(values['fps']);time=np.arange(len(j))/fps;duration=float(time[-1])
        assert j.shape[1]==23 and np.isfinite(j).all() and np.isfinite(pos).all()
        rot=Rotation.from_rotvec(aa[:,0]);quat=rot.as_quat()[:,[3,0,1,2]]
        source_quat=values['root_rot']
        xyzw_error=np.minimum(np.linalg.norm(source_quat-rot.as_quat(),axis=1),np.linalg.norm(source_quat+rot.as_quat(),axis=1)).max()
        assert xyzw_error<.001
        q=np.c_[pos,quat,j]
        grid=Image.new('RGB',(1280,4*264),'white');draw=ImageDraw.Draw(grid)
        for n,frame in enumerate(np.linspace(0,len(q)-1,16,dtype=int)):
            data.qpos[:]=q[frame];mujoco.mj_forward(native,data)
            camera.lookat[:]=q[frame,:3];camera.lookat[2]=.65
            renderer.update_scene(data,camera=camera)
            tile=Image.fromarray(renderer.render());x=(n%4)*320;y=(n//4)*264
            grid.paste(tile,(x,y));draw.text((x+5,y+241),f'{path.stem}  {frame/fps:.2f}s / {duration:.2f}s',fill='black')
        grid.save(OUT/'review'/f'{path.stem}.jpg',quality=90)
        sample_t=np.arange(int(np.floor(duration*50))+1)/50
        sampled=np.zeros((len(sample_t),32));sampled[:,:3]=np.stack([np.interp(sample_t,time,pos[:,k]) for k in range(3)],axis=1)
        sampled[:,:2]-=sampled[0,:2].copy()
        sampled[:,3:7]=Slerp(time,rot)(sample_t).as_quat()[:,[3,0,1,2]]
        for k,name in enumerate(controller.names):
            if name in names:sampled[:,7+k]=np.interp(sample_t,time,j[:,names.index(name)])
            else:assert 'wrist_roll' in name
        target=OUT/'references'/f'{path.stem}.npz'
        np.savez_compressed(target,**arrays(model,sampled,controller.names,bodyids))
        contact=np.asarray(values['contact_mask'])
        limits=native.jnt_range[1:]
        violation=np.maximum(limits[:,0]-j,j-limits[:,1]).clip(0)
        rows.append({'name':path.stem,'category':'combat_candidate' if path.stem in combat else 'pose_control' if 'pose' in path.stem else 'noncombat_control',
            'source_path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'upstream_source_key':source_key,
            'source_frames':len(j),'source_fps':fps,'duration_s':duration,'target_frames':len(sample_t),
            'target_fps':50,'target_reference':str(target),'target_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'resample_duration_loss_s':duration-sample_t[-1],'target_added_joints':['left_wrist_roll_joint','right_wrist_roll_joint'],
            'motion_dependencies':'Processed upstream motion; origin hints retained, original human recordings and licences not reconstructed from path names',
            'has_corresponding_bundled_policy':path.stem in ['Horse-stance_pose','Horse-stance_punch'],
            'contact_shape':list(contact.shape),'contact_unique_values':np.unique(contact).tolist(),
            'contact_label_ground_truth_verified':False,'joint_limit_excess_max_rad':float(violation.max()),
            'semantic_label_source':'upstream filename; contact sheet supplied for review',
            'independent_recording_verified':False,'assigned_split':'development_only'})
    renderer.close()
    output={'upstream':'https://github.com/TeleHuman/PBHC','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip(),
        'licence':'Repository CC-BY-NC-4.0; source-specific rights not inferred',
        'source_model':'official PBHC23DOF','target_model':'TWIST25DOF, added wrists neutral',
        'counts':{'all':len(rows),'combat_candidates':len(combat)},'motions':rows,
        'limitations':['Not a new independent test benchmark','No front-kick sample in bundled directory','Two checkpoints of same pose do not create two sources',
                       'Converted references test cross-controller compatibility, not a head-to-head algorithm score']}
    (OUT/'inventory.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print('PBHC_SEED_PANEL_READY',len(rows),flush=True)

if __name__=='__main__':main()
