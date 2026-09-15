"""Seeded numerical witness for geometry QP; not a learned policy experiment."""
import json
import numpy as np
import mujoco
from combat.twist_tracking import ROOT,scene,official_contract
from combat.paired_refinement import refine,collision_rows,penetration

model,_=scene(2);_,scope=official_contract();default=scope['DEFAULT_MIMIC_OBS']['g1']
q=np.r_[0,0,default[0]-.08,1,0,0,0,default[8:],0,1.5,default[0]-.08,1,0,0,0,default[8:]]
params=json.loads((ROOT/'configs/paired_refinement_v1.json').read_text(encoding='utf-8'))
raw=np.tile(q,(4,1));fixed,audit=refine(model,raw,50,params)
before=float(audit['penetration_before'][:,0].max());after=float(audit['penetration_after'][:,0].max())
assert before>.04 and after<before*.25,(before,after)
data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_forward(model,data)
rows,targets=collision_rows(model,data,.001)
assert any(abs(r[2]-1)<1e-7 for r in rows),'Ground normal/Jacobian sign mismatch'
assert np.isfinite(fixed).all()
print('PAIRED_QP_WITNESS',json.dumps({'frames':4,'ground_before_m':before,'ground_after_m':after,'normal_jacobian_verified':True,'dynamics_certified':False}))
