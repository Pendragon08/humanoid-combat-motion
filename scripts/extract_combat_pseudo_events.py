"""Conservative geometric contact candidates, never manual or force ground truth.

Uses only official paired positions and pre-existing split/tag metadata. The
selected moving limb defines a provisional actor, not a proven attack intention.
Unknown/ambiguous sources remain unlabeled. No opponent tactics are inferred.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from combat.interaction import ContactEvent,evaluate_event
from combat.protocol import load_manifest


def candidates(x,tags,fps=30.):
    radii=np.full(22,.055);radii[9]=.15;radii[15]=.10
    radii[[20,21]]=.05;radii[[10,11]]=.07
    result=[];limbs=[]
    if any(tag in tags for tag in ('punch','boxing','martial_arts')):limbs += [20,21]
    if any(tag in tags for tag in ('kick','martial_arts')):limbs += [10,11]
    for actor in (0,1):
        other=1-actor
        for limb in limbs:
            relative=x[:,actor,limb]-x[:,actor,0]
            speed=np.linalg.norm(np.diff(relative,axis=0),axis=-1)*fps
            for target in (9,15):
                gap=np.linalg.norm(x[:,actor,limb]-x[:,other,target],axis=-1)-radii[limb]-radii[target]
                for t in range(4,len(x)-4):
                    if gap[t]>gap[t-1] or gap[t]>gap[t+1]:continue
                    closing=(gap[t-3]-gap[t])*fps/3
                    if closing<.5 or np.max(speed[t-3:t])<.5:continue
                    start=max(0,t-3);end=min(len(x)-1,t+3)
                    event=ContactEvent('hit',start,end,limb,target,a_actor=actor,b_actor=other,
                        confidence=.5,provenance='automatic_sphere_geometry_and_weak_text_candidate; unreviewed')
                    metric=evaluate_event(x,radii,fps,event,tolerance=.03)
                    if metric['success']:
                        result.append((gap[t],event,metric,closing))
            # A putative block must preserve torso clearance and have an incoming
            # limb aimed toward the torso. It is still an unreviewed candidate.
            for guard in (18,19,20,21):
                gap=np.linalg.norm(x[:,actor,limb]-x[:,other,guard],axis=-1)-radii[limb]-radii[guard]
                t=int(np.argmin(gap))
                if t<4 or t>=len(x)-4:continue
                v=(x[t,actor,limb]-x[t-3,actor,limb])*fps/3
                direction=x[t-3,other,9]-x[t-3,actor,limb]
                aimed=float(np.dot(v,direction)/(np.linalg.norm(v)*np.linalg.norm(direction)+1e-9))
                if aimed<.7 or np.linalg.norm(v)<.5:continue
                event=ContactEvent('block',t-3,t+3,limb,guard,a_actor=actor,b_actor=other,protected_joint=9,
                    confidence=.4,provenance='automatic_guard_geometry_candidate; unreviewed')
                metric=evaluate_event(x,radii,fps,event)
                if metric['success']:result.append((gap[t],event,metric,float(np.linalg.norm(v))))
    # Select strongest geometric proximity, not based on downstream tracking.
    result.sort(key=lambda r:r[0])
    return result[:1],radii


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=[];counts={};examined={}
    for record in load_manifest(a.manifest):
        if not record.get('weak_tags'):continue
        split=record['split'];examined[split]=examined.get(split,0)+1
        x=np.stack([np.load(path,allow_pickle=False)[:,:66].reshape(-1,22,3) for path in record['motion_paths']],axis=1)
        hits,radii=candidates(x,record['weak_tags'])
        for distance,event,metric,speed in hits:
            rows.append({'sample_id':record['sample_id'],'source_ids':record['source_ids'],'split':split,
                'event':asdict(event),'metric':metric,'closing_speed_proxy_m_s':speed,'radii_m':radii.tolist(),
                'fps_assumption':30,'coordinates':'original paired world positions; no independent recentering',
                'label_quality':'unreviewed_kinematic_pseudo_event','semantic_attack_role_verified':False,
                'contact_force_verified':False})
            key=split+'_'+event.kind;counts[key]=counts.get(key,0)+1
    a.output.mkdir(parents=True)
    (a.output/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary={'examined':examined,'counts':counts,'events':len(rows),
        'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        'avoid_events':'not inferred: accurate counterfactual threat labels unavailable',
        'purpose':'candidate mining for review and conditional-model pilot; not contact/causal ground truth'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
