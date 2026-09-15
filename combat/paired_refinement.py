"""Local sequential QP geometry correction in the exact TWIST collision model.

Both actors stay in one world. No time scaling, actor separation, collision mask
editing, or pose replacement. Soft constraints may remain unsatisfied; audits
report actual FK contact penetration instead of optimizer residuals.
"""
import numpy as np
import mujoco
import osqp
from scipy import sparse

def jac(model,data,point,body):
    j=np.zeros((3,model.nv));mujoco.mj_jac(model,data,j,None,point,int(body));return j

def collision_rows(model,data,clearance):
    rows=[];targets=[]
    for k in range(data.ncon):
        c=data.contact[k]
        if c.dist>=clearance:continue
        b1=model.geom_bodyid[c.geom1];b2=model.geom_bodyid[c.geom2]
        row=c.frame[:3]@(jac(model,data,c.pos,b2)-jac(model,data,c.pos,b1))
        if np.linalg.norm(row)>1e-8:rows.append(row);targets.append(clearance-float(c.dist))
    return rows,targets

def penetration(model,data):
    by={'ground':0.,'self':0.,'interrobot':0.}
    for c in data.contact:
        b1=model.geom_bodyid[c.geom1];b2=model.geom_bodyid[c.geom2]
        n1=model.body(b1).name;n2=model.body(b2).name
        kind='ground' if b1==0 or b2==0 else ('interrobot' if n1[:3]!=n2[:3] else 'self')
        by[kind]=max(by[kind],-float(c.dist))
    return [by[k] for k in ('ground','self','interrobot')]

def sole(model,data,body):
    points=[]
    for g in range(model.ngeom):
        if model.geom_bodyid[g]!=body or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE:
            point=data.geom_xpos[g].copy();point[2]-=model.geom_size[g,0];points.append(point);continue
        mesh=model.geom_dataid[g]
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
        v=model.mesh_vert[start:start+count]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        points.append(v[np.argmin(v[:,2])])
    if not points:raise ValueError('No supported collision foot geometry')
    return min(points,key=lambda v:v[2])

def difference(model,a,b):
    v=np.zeros(model.nv);mujoco.mj_differentiatePos(model,v,1.,a,b);return v

def objective(model,data,original,previous,weights,params,feet,mask,anchors):
    correction=difference(model,original,data.qpos)
    value=float(np.sum(weights*correction**2)+params['correction_temporal_weight']*np.sum((correction-previous)**2))
    value+=params['collision_weight']*sum(max(params['clearance_m']-float(c.dist),0.)**2 for c in data.contact)
    for f,b in enumerate(feet):
        if mask[f]:
            value+=params['support_z_weight']*(sole(model,data,b)[2]-params['clearance_m'])**2
            value+=params['support_xy_weight']*np.sum((data.xpos[b,:2]-anchors[f,:2])**2)
    return float(value)

def support_schedule(model,raw,fps,params,feet):
    data=mujoco.MjData(model);centers=[];heights=[]
    for q in raw:
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        centers.append(data.xpos[feet].copy());heights.append([sole(model,data,b)[2] for b in feet])
    centers=np.asarray(centers);heights=np.asarray(heights)
    speed=np.linalg.norm(np.gradient(centers,1/fps,axis=0)[:,:,:2],axis=2)
    mask=np.zeros_like(heights,dtype=bool)
    for actor in range(2):
        ix=slice(2*actor,2*actor+2)
        mask[:,ix]=(heights[:,ix]-heights[:,ix].min(axis=1,keepdims=True)<params['support_relative_height_m']) & (speed[:,ix]<params['support_speed_threshold_m_s'])
    anchors=centers.copy()
    for f in range(4):
        start=0
        for t in range(len(raw)):
            if mask[t,f] and (t==0 or not mask[t-1,f]):start=t
            if mask[t,f]:anchors[t,f,:2]=centers[start,f,:2]
    return mask,anchors

def refine(model,raw,fps,params):
    data=mujoco.MjData(model)
    feet=[model.body(f'a{i}_{s}_ankle_roll_link').id for i in range(2) for s in ('left','right')]
    mask,anchors=support_schedule(model,raw,fps,params,feet)
    weights=np.tile(np.r_[params['root_xy_weight'],params['root_xy_weight'],params['root_z_weight'],
        np.full(3,params['root_rotation_weight']),np.full(25,params['posture_joint_weight'])],2)
    step=np.tile(np.r_[np.full(3,.025),np.full(3,.07),np.full(25,.12)],2)
    maxchange=np.tile(np.r_[np.full(2,params['max_root_xy_change_m']),params['max_root_z_change_m'],
        np.full(3,params['max_root_rotation_change_rad']),np.full(25,params['max_joint_change_rad'])],2)
    output=[];before=[];after=[];solver_status=[];previous_correction=np.zeros(model.nv)
    for t,original in enumerate(raw):
        data.qpos[:]=original;mujoco.mj_forward(model,data);before.append(penetration(model,data))
        mujoco.mj_integratePos(model,data.qpos,previous_correction,1.)
        # A previous frame correction can put the NEW frame outside its joint
        # limits. Project that warm start before imposing the local trust region.
        for j in range(model.njnt):
            if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:
                qi=model.jnt_qposadr[j]
                data.qpos[qi]=np.clip(data.qpos[qi],*model.jnt_range[j])
        for iteration in range(params['iterations']):
            mujoco.mj_forward(model,data)
            back=difference(model,data.qpos,original)
            rows=[np.eye(model.nv)];goals=[back];w=[weights]
            rows.append(np.eye(model.nv));goals.append(back+previous_correction);w.append(np.full(model.nv,params['correction_temporal_weight']))
            cr,ct=collision_rows(model,data,params['clearance_m'])
            if cr:rows.append(np.array(cr));goals.append(np.array(ct));w.append(np.full(len(cr),params['collision_weight']))
            for f,b in enumerate(feet):
                if not mask[t,f]:continue
                bottom=sole(model,data,b)
                rows.append(jac(model,data,bottom,b)[2:3]);goals.append(np.array([params['clearance_m']-bottom[2]]));w.append(np.array([params['support_z_weight']]))
                rows.append(jac(model,data,data.xpos[b],b)[:2]);goals.append(anchors[t,f,:2]-data.xpos[b,:2]);w.append(np.full(2,params['support_xy_weight']))
            J=np.vstack(rows);goal=np.concatenate(goals);W=np.concatenate(w)
            H=J.T@(W[:,None]*J)+np.eye(model.nv)*1e-6;g=-J.T@(W*goal)
            correction=difference(model,original,data.qpos)
            lower=np.maximum(-step,-maxchange-correction);upper=np.minimum(step,maxchange-correction)
            for j in range(model.njnt):
                if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
                vi=model.jnt_dofadr[j];qi=model.jnt_qposadr[j]
                lower[vi]=max(lower[vi],model.jnt_range[j,0]-data.qpos[qi])
                upper[vi]=min(upper[vi],model.jnt_range[j,1]-data.qpos[qi])
            if np.any(lower>upper+1e-8):
                invalid=np.flatnonzero(lower>upper+1e-8)
                raise ValueError(f'Bounds incompatible frame={t}, iteration={iteration}, dofs={invalid.tolist()}, lower={lower[invalid].tolist()}, upper={upper[invalid].tolist()}')
            qp=osqp.OSQP();qp.setup(P=sparse.csc_matrix(np.triu(H)),q=g,A=sparse.eye(model.nv,format='csc'),
                l=lower,u=upper,verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=1000,polishing=False)
            sol=qp.solve(raise_error=False);solver_status.append(sol.info.status)
            if sol.info.status_val not in (1,2):raise RuntimeError(sol.info.status)
            current=data.qpos.copy()
            best=objective(model,data,original,previous_correction,weights,params,feet,mask[t],anchors[t])
            bestq=current.copy()
            for scale in (1.,.5,.25,.125):
                data.qpos[:]=current;mujoco.mj_integratePos(model,data.qpos,sol.x,scale);mujoco.mj_forward(model,data)
                score=objective(model,data,original,previous_correction,weights,params,feet,mask[t],anchors[t])
                if score<best:best=score;bestq=data.qpos.copy()
            data.qpos[:]=bestq
        mujoco.mj_forward(model,data);after.append(penetration(model,data));output.append(data.qpos.copy())
        previous_correction=difference(model,original,data.qpos)
    return np.asarray(output),{'penetration_before':np.array(before),'penetration_after':np.array(after),
        'support_mask':mask,'support_anchors':anchors,'solver_status':solver_status}
