"""Paired reference ablation and separately labeled, unassisted physical rollouts."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from combat.diffusion import MotionDenoiser, augment
from combat.robot import ROOT, Robot
from combat.constraints import project_joints, correct_foot_contact, metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--per-clip',type=int,default=4);p.add_argument('--variants',type=int,default=2);p.add_argument('--noise-step',type=int,default=40)
    args=p.parse_args();out=ROOT/'runs'/args.run;out.mkdir(exist_ok=True)
    torch.manual_seed(171)
    # Only locally trained checkpoint; do not unpickle arbitrary downloaded weights.
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model=MotionDenoiser().cuda();model.load_state_dict(checkpoint['model']);model.eval()
    mean=checkpoint['mean'].cuda();std=checkpoint['std'].cuda()
    z=np.load(ROOT/'data/windows.npz',allow_pickle=False)
    selected=[]
    for clip in sorted(set(z['test_ids'])):
        ids=np.flatnonzero(z['test_ids']==clip)
        selected.extend(ids[np.linspace(0,len(ids)-1,min(args.per_clip,len(ids))).astype(int)].tolist())
    indices=np.repeat(selected,args.variants)
    seeds=z['test'][indices].astype(np.float64)
    root=torch.tensor(seeds[:,:,:7],dtype=torch.float32,device='cuda')
    style=torch.tensor(z['test_style'][indices],dtype=torch.long,device='cuda')
    x=(torch.tensor(seeds[:,:,7:],dtype=torch.float32,device='cuda')-mean)/std
    generated=augment(model,x,root,style,noise_step=args.noise_step,random_seed=500+checkpoint['config']['seed'])
    generic=seeds.copy();generic[:,:,7:]=(generated*std+mean).cpu().numpy()
    alternatives={}
    for level in [20,60]:
        pred=augment(model,x,root,style,noise_step=level,random_seed=500+checkpoint['config']['seed'])
        alt=seeds.copy();alt[:,:,7:]=(pred*std+mean).cpu().numpy();alternatives[level]=alt
    robot=Robot();records=[];motions=[];candidate_audit=[];started=time.time()
    for i,(seed,candidate) in enumerate(zip(seeds,generic)):
        projected,status=project_joints(candidate,robot)
        grounded,contact=correct_foot_contact(projected,seed,robot)
        # Fixed, predeclared test-time physics feedback. Costs and rejected candidates are saved.
        # It is inference-time selection under this PD controller, not policy-generalization evidence.
        candidates=[(args.noise_step,grounded)]
        for level,alt in alternatives.items():
            jq,_=project_joints(alt[i],robot);fq,_=correct_foot_contact(jq,seed,robot);candidates.append((level,fq))
        scored=[]
        for level,q in candidates:
            check,_=robot.diagnostic_rollout(q)
            deviation=float(np.sqrt(np.mean((q[:,7:]-seed[:,7:])**2)))
            fallen_fraction=0. if check['pd_fall_time_s'] is None else max(0,1-check['pd_fall_time_s']/check['duration_s'])
            cost=check['pd_root_rmse_m']+.2*check['pd_joint_rmse_rad']+2*fallen_fraction+.1*deviation
            scored.append((cost,level,q))
            candidate_audit.append({'case':i,'noise_step':level,'selection_cost':cost,'seed_joint_deviation':deviation,**check})
        _,chosen_level,chosen=min(scored,key=lambda z:(z[0],z[1]))
        arms={'video_seed':seed,'generic_diffusion':candidate,'joint_constrained':projected,'ground_refined':grounded,'physics_selected':chosen}
        for arm,q in arms.items():
            row={'case':i,'source_id':str(z['test_ids'][indices[i]]),'source_start_frame':int(z['test_starts'][indices[i]]),'variant':i%args.variants,'arm':arm,'selected_noise_step':None}
            row.update(metrics(q,seed,robot))
            dynamic,_=robot.diagnostic_rollout(q)
            row.update(dynamic)
            if arm=='physics_selected':row['selected_noise_step']=chosen_level
            # Rejection remains visible in results; never silently remove hard cases.
            row['strict_filter_pass']=bool(row['joint_limit_violation_fraction']==0 and row['velocity_violation_fraction']==0 and row['acceleration_violation_fraction']==0 and row['self_collision_frame_fraction']==0 and row['ground_penetration_peak_m']<.02 and row['pd_full_horizon_success'])
            records.append(row)
        motions.append(np.stack(list(arms.values())))
        (out/'metrics.partial.json').write_text(json.dumps(records,indent=2))
        print('EVALUATED_CASE',i+1,'/',len(seeds),'seconds',round(time.time()-started),flush=True)
    np.savez_compressed(out/'motions.npz',qpos=np.stack(motions),arms=np.array(list(arms)),source_ids=z['test_ids'][indices],source_starts=z['test_starts'][indices],fps=np.array(30.),joint_names=np.array(robot.names),quaternion_order=np.array('wxyz'))
    (out/'metrics.json').write_text(json.dumps(records,indent=2))
    (out/'physics_candidates.json').write_text(json.dumps(candidate_audit,indent=2))
    number_keys=[k for k,v in records[0].items() if isinstance(v,(int,float,bool)) and k not in ['case','source_start_frame','variant']]
    summary={}
    for arm in arms:
        rows=[r for r in records if r['arm']==arm]
        summary[arm]={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in number_keys}
    report={'run':args.run,'training_seed':checkpoint['config']['seed'],'checkpoint_step':checkpoint['step'],'test_sources':sorted(set(z['test_ids'][indices].tolist())),'windows_per_source':args.per_clip,'variants_per_window':args.variants,'cases':len(seeds),'arms':summary,'elapsed_s':time.time()-started,'limitations':['Video-derived seeds, not true optical MoCap.','Root path is a seed condition, not newly generated.','Styles are broad martial-arts labels, no verified jab/block event labels.','Ground contact proxy only; no paired opponent or combat force.','Torque-limited PD diagnostic is not a learned imitation policy.','No evidence yet for synthetic-data benefit to policy generalization.','Test has four source clips from one held-out routine; not independent performer-level generalization.']}
    report['physics_selection']={'noise_levels':[20,args.noise_step,60],'cost':'root_rmse + 0.2*joint_rmse + 2*fraction_of_horizon_after_fall + 0.1*seed_joint_deviation','budget':'three candidate rollouts per generated case, in addition to common final evaluation','claim_boundary':'Inference-time optimization under the same diagnostic controller; not a held-out-controller generalization result.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    (out/'EVALUATION_COMPLETE.json').write_text(json.dumps({'cases':len(seeds),'seconds':time.time()-started}))
    print('EVALUATION_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
