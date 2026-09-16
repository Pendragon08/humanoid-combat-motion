"""Audit PM01 reference artifacts including independent SDK/USD FK comparison."""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[1]
def main():
    run=ROOT/'runs/engineai_single_v1';model=mujoco.MjModel.from_xml_path(str(ROOT/'cache/engineai_overlay/pm01_kinematic.xml'));data=mujoco.MjData(model)
    modelnames=[model.joint(i).name for i in range(1,model.njnt)];rows=[]
    for p in sorted((run/'references').glob('sample_*.npz')):
        with np.load(p,allow_pickle=False) as z:q=z['qpos'];fps=float(z['fps'])
        with np.load(run/'tracking_references'/p.name,allow_pickle=False) as z:
            names=z['joint_names'].tolist();bn=z['body_names'].tolist();j=z['joint_pos'];bp=z['body_pos_w'];rate=float(z['fps'])
        ts=np.arange(len(q))/fps;target=np.arange(len(j))/rate
        rootpos=np.stack([np.interp(target,ts,q[:,i]) for i in range(3)],-1)
        quat=Slerp(ts,Rotation.from_quat(q[:,[4,5,6,3]]))(target).as_quat()[:,[3,0,1,2]]
        indexes=[names.index(n) for n in modelnames];errs=[];articulated_errors=[];per_body={}
        body_pairs=[(model.body(n).id,bn.index(n)) for n in bn if n in [model.body(i).name for i in range(1,model.nbody)]]
        for t in range(len(j)):
            data.qpos[:3]=rootpos[t];data.qpos[3:7]=quat[t];data.qpos[7:]=j[t,indexes];mujoco.mj_forward(model,data)
            for m,b in body_pairs:
                error=float(np.linalg.norm(data.xpos[m]-bp[t,b]));errs.append(error)
                name=bn[b];per_body[name]=max(error,per_body.get(name,0.))
                if name not in ['LINK_FOOT_L','LINK_FOOT_R','LINK_ELBOW_END_L','LINK_ELBOW_END_R']:articulated_errors.append(error)
        row={'file':p.name,'finite':bool(np.isfinite(bp).all() and np.isfinite(j).all()),'frames':len(j),'joint_count':len(names),'sdk_usd_mean_body_discrepancy_m':float(np.mean(errs)),'sdk_usd_max_body_discrepancy_m':float(np.max(errs)),'articulated_body_max_discrepancy_m':float(np.max(articulated_errors)),'per_body_max_m':per_body}
        rows.append(row)
    result={'samples':rows,'all_12_present':len(rows)==12,'all_body_fk_agreement_under_1mm':all(x['sdk_usd_max_body_discrepancy_m']<.001 for x in rows),'articulated_body_fk_agreement_under_1mm':all(x['articulated_body_max_discrepancy_m']<.001 for x in rows),'discrepancy':'Four fixed FOOT/ELBOW_END child frames differ between SDK and USD/PhysX exposure. They are not official tracked body targets. Full reference arrays are computed in the target USD model, never copied from SDK FK. Differences retained, no all-body equivalence claim.','claim':'reference consistency check, not execution success'}
    out=ROOT/'records/engineai_single_v1/transfer_audit.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
