"""Audit whether human proxy contacts survive IK; not collision-force truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from combat.interaction import ContactEvent,evaluate_event

p=argparse.ArgumentParser();p.add_argument('--human-evaluation',type=Path,required=True)
p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
root=Path(__file__).resolve().parents[1]
model=mujoco.MjModel.from_xml_path(str(root/'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml'));data=mujoco.MjData(model)
mapping={0:'pelvis',4:'left_knee_link',5:'right_knee_link',7:'left_ankle_roll_link',8:'right_ankle_roll_link',
         9:'torso_link',18:'left_elbow_link',19:'right_elbow_link',20:'left_wrist_yaw_link',21:'right_wrist_yaw_link'}
body_ids={j:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name) for j,name in mapping.items()}
if any(i<0 for i in body_ids.values()):raise ValueError('Robot link missing')
manifest=json.loads((a.references/'manifest.json').read_text());scale=manifest['scale']
evaluation=json.loads(a.human_evaluation.read_text());rows=[]
radii=np.full(22,.055);radii[9]=.15;radii[15]=.1;radii[[20,21]]=.05;radii[[10,11]]=.07
for index,case in enumerate(evaluation['cases']):
    event=ContactEvent(**case['pseudo_event'])
    required={event.a_joint,event.b_joint}
    if event.protected_joint is not None:required.add(event.protected_joint)
    if not required<=body_ids.keys():
        rows.append({'source':case['source'],'evaluated':False,'reason':'Unmapped human contact joint','joints':sorted(required)});continue
    path=a.references/f'sample_{index:03d}.npz'
    with np.load(path,allow_pickle=False) as z:q=z['qpos'];fps=float(z['fps'])
    proxy=np.zeros((len(q),2,22,3))
    for t in range(len(q)):
        for actor in (0,1):
            data.qpos[:]=q[t,actor];mujoco.mj_forward(model,data)
            for j,b in body_ids.items():proxy[t,actor,j]=data.xpos[b]/scale
    result=evaluate_event(proxy,radii,fps,event)
    rows.append({'source':case['source'],'evaluated':True,'contact':result})
valid=[r for r in rows if r['evaluated']]
out={'cases':rows,'mapped_cases':len(valid),'proxy_success_fraction':np.mean([r['contact']['success'] for r in valid]).item() if valid else None,
     'mapping':mapping,'scale':scale,'physics_verified':False,
     'scope':'human sphere radii scaled to robot links; tests geometric preservation, not robot collision meshes or contact forces'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='cases'}),flush=True)
