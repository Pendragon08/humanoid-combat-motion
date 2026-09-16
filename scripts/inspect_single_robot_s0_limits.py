"""Read-only reference constraints audit; not dynamic feasibility certification."""
import json
import numpy as np
from combat.twist_tracking import ROOT, scene, Controller
from scripts.single_robot_s0 import OUT, save_json

model,_=scene(1)
ctrl=Controller(model,0,None,'cpu')
protocol=json.loads((OUT/'protocol.json').read_text())
rows=[]
for source in protocol['sources']:
    with np.load(OUT/'references'/f'{source["id"]}_full.npz') as z:
        q=z['joint_pos'];v=z['joint_vel']
    limits=model.jnt_range[ctrl.jids]
    violations=np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0)
    joints=[]
    for j,name in enumerate(ctrl.names):
        if violations[:,j].max()>1e-6:
            joints.append({'name':name,'lower_rad':float(limits[j,0]),'upper_rad':float(limits[j,1]),
                'observed_min_rad':float(q[:,j].min()),'observed_max_rad':float(q[:,j].max()),
                'max_excess_rad':float(violations[:,j].max()),
                'fraction_above_0p01_rad':float((violations[:,j]>.01).mean())})
    rows.append({'id':source['id'],'joints':joints,
        'any_violation_above_0p01_rad_frame_fraction':float(np.any(violations>.01,axis=1).mean()),
        'peak_joint_speed_rad_s':float(abs(v).max())})
save_json(OUT/'limits_audit.json',{'tolerance_rad':.01,'target_model':'official TWIST25DOF',
    'source_model':'GMR29DOF; mapped by joint name',
    'scope':'Geometric range comparison; no torque feasibility conclusion','motions':rows})
print(json.dumps(rows,ensure_ascii=False))
