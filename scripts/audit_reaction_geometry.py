"""Audit saved contact-refinement outputs without rerunning or changing them."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.run/'evaluation.json').read_text());rows=[]
    chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
    edges=np.array([edge for chain in chains for edge in zip(chain[:-1],chain[1:])]);parent,child=edges.T
    for i,case in enumerate(report['cases']):
        old=np.load(a.run/'actor_conditioned'/f'sample_{i:03d}.npz')['positions']
        new=np.load(a.run/'actor_conditioned_contact'/f'sample_{i:03d}.npz')['positions']
        change=np.abs(np.linalg.norm(old[:,:,parent]-old[:,:,child],axis=-1)-np.linalg.norm(new[:,:,parent]-new[:,:,child],axis=-1))
        maximum=float(change.max());contact=case['arms']['actor_conditioned_contact']['contact']['success']
        rows.append({'source':case['source'],'max_bone_length_change_m':maximum,
                     'p95_bone_length_change_m':float(np.percentile(change,95)),
                     'max_defender_joint_speed_m_s':float(np.linalg.norm(np.diff(new[:,1],axis=0)*30,axis=-1).max()),
                     'known_actor_max_displacement_m':float(np.abs(new[:,0]-old[:,0]).max()),
                     'min_defender_joint_height_m':float(new[:,1,:,1].min()),
                     'proxy_contact_success':contact,'contact_and_bone_change_below_2cm':contact and maximum<=.02})
    summary={arm:{'proxy_contact_success_fraction':float(np.mean([c['arms'][arm]['contact']['success'] for c in report['cases']])),
                  'defender_position_rmse_m':float(np.mean([c['arms'][arm]['defender_position_rmse_m'] for c in report['cases']]))}
             for arm in ('inpainting_only','actor_conditioned','actor_conditioned_contact')}
    result={'cases':rows,'summary':summary,'strict_contact_and_bone_fraction':float(np.mean([r['contact_and_bone_change_below_2cm'] for r in rows])),
            'maximum_bone_change_m':max(r['max_bone_length_change_m'] for r in rows),
            'bone_tolerance':'2cm is a declared research screening threshold, not a robot specification',
            'physics_validated':False,'caveat':'Optimized geometric contact alone can be achieved by deforming bones; this check detects that failure.'}
    (a.run/'geometry_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
