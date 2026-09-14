"""Re-evaluate contact projection on saved outputs; never retrain on test labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from combat.contact_guidance import refine_contacts
from combat.contact_kinematics import refine_kinematic_contacts
from combat.interaction import ContactEvent,evaluate_event


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--method',choices=['projection','kinematic'],default='projection')
    p.add_argument('--contact-target',type=float,default=0.,help='Kinematic target gap;0.03 reproduces boundary-target ablation')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=json.loads((a.run/'evaluation.json').read_text());a.output.mkdir(parents=True)
    chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
    edges=[edge for chain in chains for edge in zip(chain[:-1],chain[1:])];parent,child=np.array(edges).T
    radii=np.full(22,.055);radii[9]=.15;radii[15]=.1;radii[[20,21]]=.05;radii[[10,11]]=.07
    rows=[]
    for index,case in enumerate(report['cases']):
        path=a.run/'actor_conditioned'/f'sample_{index:03d}.npz'
        x=np.load(path)['positions'];event=ContactEvent(**case['pseudo_event'])
        if a.method=='kinematic':
            y,history=refine_kinematic_contacts(x,radii,30,[event],edges,steps=200,device=a.device,fixed_actor=0,contact_target=a.contact_target)
        else:
            y,history=refine_contacts(x,radii,30,[event],edges,steps=80,device=a.device,fixed_actor=0,hard_bone_projection=True)
        bone_delta=np.abs(np.linalg.norm(y[:,:,parent]-y[:,:,child],axis=-1)-np.linalg.norm(x[:,:,parent]-x[:,:,child],axis=-1))
        metric=evaluate_event(y,radii,30,event)
        np.savez_compressed(a.output/f'sample_{index:03d}.npz',positions=y,fps=30)
        rows.append({'source':case['source'],'contact':metric,'max_bone_length_change_m':float(bone_delta.max()),
            'known_actor_displacement_m':float(np.abs(y[:,0]-x[:,0]).max()),
            'max_defender_joint_speed_m_s':float(np.linalg.norm(np.diff(y[:,1],axis=0)*30,axis=-1).max()),
            'max_displacement_m':float(np.linalg.norm(y-x,axis=-1).max()),'final_loss':history[-1]})
    result={'cases':rows,'contact_success_fraction':float(np.mean([r['contact']['success'] for r in rows])),
        'maximum_bone_change_m':max(r['max_bone_length_change_m'] for r in rows),
        'method':a.method+' contact optimization with per-frame source bone lengths and fixed actor0',
        'optimization_surface_gap_target_m':a.contact_target if a.method=='kinematic' else .03,
        'status':'exploratory post-hoc diagnostic; not preregistered confirmatory test',
        'physics_validated':False,'morphology_and_dynamics_guaranteed':False}
    (a.output/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
