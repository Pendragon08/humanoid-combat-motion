"""Independent URDF FK + support-label checks; no physics or success relabeling."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import joblib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_kinematic_audit_20260917'
URDF=ROOT/'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf'
tree=ET.parse(URDF).getroot()


def xyz(node,attr='xyz'):
    return np.fromstring(node.get(attr,'0 0 0'),sep=' ') if node is not None else np.zeros(3)


def fk(q,names):
    links={n.get('name') for n in tree.findall('link')}
    joints=tree.findall('joint');roots=links-{j.find('child').get('link') for j in joints}
    assert len(roots)==1,roots
    root=next(iter(roots));assert root=='pelvis',root
    transforms={root:np.tile(np.eye(4),(len(q),1,1))}
    transforms[root][:,:3,:3]=Rotation.from_quat(q[:,3:7][:,[1,2,3,0]]).as_matrix()
    transforms[root][:,:3,3]=q[:,:3]
    pending=joints.copy()
    while pending:
        progress=False
        for joint in pending[:]:
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in transforms:continue
            origin=joint.find('origin');tf=np.eye(4);tf[:3,3]=xyz(origin)
            tf[:3,:3]=Rotation.from_euler('xyz',xyz(origin,'rpy')).as_matrix()
            motion=np.tile(np.eye(4),(len(q),1,1));kind=joint.get('type')
            if kind!='fixed':
                assert kind in ['revolute','continuous'],kind
                ix=names.index(joint.get('name'));axis=xyz(joint.find('axis'))
                motion[:,:3,:3]=Rotation.from_rotvec(q[:,7+ix,None]*axis[None]).as_matrix()
            transforms[child]=transforms[parent]@tf@motion
            pending.remove(joint);progress=True
        if not progress:raise ValueError('Disconnected URDF')
    return transforms


def center_of_mass(transforms):
    total=0.;weighted=0.
    for link in tree.findall('link'):
        inertial=link.find('inertial')
        if inertial is None:continue
        mass=float(inertial.find('mass').get('value'))
        point=xyz(inertial.find('origin'));tf=transforms[link.get('name')]
        weighted=weighted+mass*(tf[:,:3,:3]@point+tf[:,:3,3]);total+=mass
    return weighted/total,total


def main():
    if REC.exists():raise FileExistsError(REC)
    inventory=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())
    windows=json.loads((ROOT/'runs/pbhc_refinement_pilot_20260916/protocol.json').read_text())['source_windows']
    rows=[]
    for case in inventory['motions']:
        if case['category']!='combat_candidate':continue
        name=case['name'];path=ROOT/'runs/pbhc_beyondmimic_20260916/references_full'/('sample_'+name+'_actor0.npz')
        with np.load(path,allow_pickle=False) as z:
            body_names=list(z['body_names']);names=list(z['joint_names']);pelvis=body_names.index('pelvis')
            q=np.c_[z['body_pos_w'][:,pelvis],z['body_quat_w'][:,pelvis],z['joint_pos']]
            transforms=fk(q,names)
            positions=np.stack([transforms[n][:,:3,3] for n in body_names],axis=1)
            errors=np.linalg.norm(positions-z['body_pos_w'],axis=-1)
            assert errors.max()<1e-4,(name,float(errors.max()))
            contacts=next(iter(joblib.load(case['source_path']).values()))['contact_mask']
            idx=np.minimum(np.rint(np.arange(len(q))/50*case['source_fps']).astype(int),len(contacts)-1)
            contacts=np.asarray(contacts)[idx]
            cm,mass=center_of_mass(transforms)
            foot=np.stack([transforms[f'{side}_ankle_roll_link'][:,:3,3] for side in ['left','right']],axis=1)
            ankle_speed=np.linalg.norm(np.diff(foot[:,:,:2],axis=0)*50,axis=-1)
            stable=np.isclose(contacts[1:],1)&np.isclose(contacts[:-1],1)
            prep=(np.arange(1,len(q))/50)<windows[name][0]
            pairs=stable.all(1)
            baseline_distance=np.linalg.norm(foot[:,0,:2]-foot[:,1,:2],axis=-1)
            distance_change=np.abs(np.diff(baseline_distance))*50
            limit_excess=0.
            for j in tree.findall('joint'):
                if j.get('name') in names and j.find('limit') is not None:
                    values=q[:,7+names.index(j.get('name'))];lim=j.find('limit')
                    limit_excess=max(limit_excess,float(np.maximum(float(lim.get('lower'))-values,values-float(lim.get('upper'))).clip(0).max()))
            metrics=dict(name=name,reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frames=len(q),fk_max_error_m=float(errors.max()),urdf_mass_kg=mass,
                reference_joint_limit_excess_rad=limit_excess,
                support_labels_verified_as_force_ground_truth=False,
                preparation_window_s=[0,windows[name][0]],
                known_contact_fraction=float(np.isclose(contacts,1).mean()),
                unknown_contact_fraction=float(np.isclose(contacts,.5).mean()),
                known_contact_ankle_xy_speed_mean_m_s=float(ankle_speed[stable].mean()) if stable.any() else None,
                known_contact_ankle_xy_speed_p95_m_s=float(np.percentile(ankle_speed[stable],95)) if stable.any() else None,
                preparation_known_contact_speed_p95_m_s=float(np.percentile(ankle_speed[stable&prep[:,None]],95)) if (stable&prep[:,None]).any() else None,
                double_support_distance_change_p95_m_s=float(np.percentile(distance_change[pairs],95)) if pairs.any() else None,
                first_reference_root_speed_m_s=float(np.linalg.norm(z['body_lin_vel_w'][0,pelvis])),
                reference_initial_com_world=cm[0].tolist(),reference_initial_ankles_world=foot[0].tolist())
        for updates in [1000,1999]:
            folder=ROOT/'runs/pbhc_followup_20260916/evaluation'/f'{name}_nominal_model_{updates}'
            with np.load(folder/'rollout.npz',allow_pickle=False) as z:
                valid=z['alive'][:,0].astype(bool);actual=z['qpos'][valid];reference=z['reference'][valid]
                tf=fk(actual,list(z['joint_names']));rf=fk(reference,list(z['joint_names']))
                root_delta=actual[:,:3]-reference[:,:3]
                actualfeet=np.stack([tf[f'{s}_ankle_roll_link'][:,:3,3] for s in ['left','right']],axis=1)
                reffeet=np.stack([rf[f'{s}_ankle_roll_link'][:,:3,3] for s in ['left','right']],axis=1)
                metrics[f'nominal_{updates+1}']=dict(valid_seconds=len(actual)/50,
                    root_xy_error_final_m=float(np.linalg.norm(root_delta[-1,:2])),
                    root_z_error_final_m=float(abs(root_delta[-1,2])),
                    root_xy_error_max_m=float(np.linalg.norm(root_delta[:,:2],axis=-1).max()),
                    ankle_relative_to_root_error_mean_m=float(np.linalg.norm((actualfeet-actual[:,None,:3])-(reffeet-reference[:,None,:3]),axis=-1).mean()),
                    first_root_error_over15cm_s=next(((i+1)/50 for i,e in enumerate(np.linalg.norm(root_delta,axis=-1)) if e>=.15),None))
        rows.append(metrics)
    REC.mkdir(parents=True)
    output={'scope':'Independent kinematic audit, no force/dynamic feasibility claim',
            'urdf_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations':['Ankle link is not a no-slip contact point; foot rolling can move it.',
                'PBHC contact0.5 is unknown and excluded; contact1 is upstream heuristic, not measured force.',
                'Preparation windows are coarse development annotations, not independent expert labels.',
                'No whole-body balance or contact-force certification; no automatic relabeling of strict pass.'], 'rows':rows}
    (REC/'audit.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))


if __name__=='__main__':main()
