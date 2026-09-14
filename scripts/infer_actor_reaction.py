"""Compare actor-clamped DDIM, trained motion conditioning and contact refinement.

Held-out full actor motion and externally supplied pseudo contact goals are inputs.
This is an offline conditional reconstruction experiment, not real-time causality.
"""
import argparse
from dataclasses import replace
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from combat.annotations import read_annotation
from combat.checkpoint_io import load_model_state
from combat.protocol import load_manifest
from combat.reaction import ActorMotionAdapter,history_condition
from combat.interaction import ContactEvent,evaluate_event
from combat.contact_guidance import refine_contacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=20)
    p.add_argument('--seed',type=int,default=123)
    a=p.parse_args();out=a.output.resolve();adapter_path=a.adapter.resolve()
    if out.exists():raise FileExistsError(out)
    records={r['sample_id']:r for r in load_manifest(a.manifest.resolve())}
    events=[json.loads(line) for line in a.events.resolve().read_text().splitlines() if line]
    cases=sorted([r for r in events if r['split']=='test'],key=lambda r:r['sample_id'])[:a.limit]
    root=Path(__file__).resolve().parents[1];import clip
    clip.load=functools.partial(clip.load,download_root=str(root/'cache/clip'));os.chdir(root/'third_party/TIMotion')
    from configs import get_config
    from models import TIMotion
    from models.cfg_sampler import ClassifierFreeSampleModel
    from models.gaussian_diffusion import MotionDiffusion,space_timesteps,ModelMeanType,ModelVarType,LossType
    from utils.preprocess import load_motion
    from utils.utils import process_motion_np,qmul_np,qinv_np,qrot_np,rigid_transform,trans_matrix
    cfg=get_config('configs/model.yaml');cfg.NUM_HEADS=16;cfg.NUM_LAYERS=5;cfg.LATENT_DIM=512;cfg.LPA=False
    model=TIMotion(cfg);state,sha=load_model_state(root/'data/official/timotion_weights.npz')
    model.load_state_dict(state,strict=True);del state;model=model.cuda().eval()
    checkpoint=torch.load(adapter_path,map_location='cpu',weights_only=False)
    history_only=checkpoint['provenance'].get('history_only',False)
    if checkpoint['provenance']['backbone_sha256']!=sha:raise ValueError('Backbone mismatch')
    training=set(checkpoint['provenance']['source_ids']['train']);validation=set(checkpoint['provenance']['source_ids']['val'])
    if any(c['sample_id'] in training|validation for c in cases):raise ValueError('Test-source leakage')
    adapter=ActorMotionAdapter().cuda().eval();adapter.load_state_dict(checkpoint['adapter'],strict=True)
    decoder=model.decoder;normalizer=decoder.diffusion.normalizer
    diffusion=MotionDiffusion(use_timesteps=space_timesteps(decoder.diffusion_steps,decoder.sampling_strategy),
        betas=decoder.betas,motion_rep=decoder.motion_rep,model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,loss_type=LossType.MSE,rescale_timesteps=False)
    guidance=ClassifierFreeSampleModel(decoder.net,decoder.cfg_weight)
    chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
    edges=[edge for chain in chains for edge in zip(chain[:-1],chain[1:])]
    frames=checkpoint['provenance']['frames'];out.mkdir(parents=True);results=[]
    for index,item in enumerate(cases):
        record=records[item['sample_id']];order=[item['event']['a_actor'],item['event']['b_actor']]
        arrays=[load_motion(record['motion_paths'][i],16,False)[0] for i in order]
        center=(item['event']['start']+item['event']['end'])//2;start=max(0,min(center-frames//2,len(arrays[0])-frames-1))
        arrays=[x[start:start+frames+1].copy() for x in arrays]
        first,qa,pa=process_motion_np(arrays[0],.001,0,22);second,qb,pb=process_motion_np(arrays[1],.001,0,22)
        relative=qmul_np(qb,qinv_np(qa));angle=np.arctan2(relative[:,2:3],relative[:,0:1]);xz=qrot_np(qa,pb-pa)[:,[0,2]]
        second=rigid_transform(np.concatenate([angle,xz],axis=-1)[0],second)
        truth=np.stack([first,second],axis=1).astype(np.float32);length=len(truth)
        cutoff=min(30,length//2)
        if history_only:
            transformed=[np.einsum('mn,tjn->tjm',trans_matrix,x[:,:66].reshape(-1,22,3)) for x in arrays]
            shared_floor=min(x[:cutoff,:,1].min() for x in transformed)
            for actor,x in enumerate(transformed):
                truth[:,actor,1:66:3]+=x[:,:,1].min()-shared_floor
                for k,j in enumerate((7,10,8,11)):
                    pos=truth[:,actor,j*3:j*3+3];vel=truth[:,actor,66+j*3:66+j*3+3]
                    truth[:,actor,258+k]=((vel**2).sum(-1)<.001)&(pos[:,1]<(0.12 if k%2==0 else .05))
        raw=torch.tensor(truth[None],device='cuda');normalized=normalizer.forward(raw).reshape(1,length,524)
        known=history_condition(raw[:,:,0],torch.tensor([cutoff],device='cuda'),normalizer) if history_only else normalized[...,:262].clone()
        observed=cutoff if history_only else length
        caption=read_annotation(record['annotation_path'])[0].splitlines()[0].strip()
        event=replace(ContactEvent(**item['event']),a_actor=0,b_actor=1,start=max(0,item['event']['start']-start),end=min(length-1,item['event']['end']-start))
        radii=np.array(item['radii_m']);truth_positions=truth[...,:66].reshape(length,2,22,3)
        row={'source':item['sample_id'],'seed':a.seed+index,'pseudo_event':event.__dict__,
             'ground_truth_geometry_after_preprocessing':evaluate_event(truth_positions,radii,30,event),'arms':{}}
        for arm in ('inpainting_only','actor_conditioned'):
            torch.manual_seed(a.seed+index);np.random.seed(a.seed+index);random.seed(a.seed+index)
            with torch.no_grad():
                text=model.text_process({'text':[caption]})['cond']
                cond=text if arm=='inpainting_only' else adapter(text,known,torch.tensor([observed],device='cuda'),
                    torch.tensor([1 if event.kind=='hit' else 2],device='cuda'),
                    torch.tensor([[event.start/length,event.end/length]],device='cuda'))
                sample=torch.randn(1,length,524,device='cuda');known_noise=torch.randn_like(known)
                for step in reversed(range(diffusion.num_timesteps)):
                    t=torch.tensor([step],device='cuda')
                    sample[:,:observed,:262]=diffusion.q_sample(known,t,noise=known_noise)[:,:observed]
                    sample=diffusion.ddim_sample(guidance,sample,t,clip_denoised=False,model_kwargs={'cond':cond,'mask':None})['sample']
                sample[:,:observed,:262]=known[:,:observed]
                generated=normalizer.backward(sample.reshape(1,length,2,262))[0].cpu().numpy()
            generated[:observed,0]=truth[:observed,0]
            if history_only:
                generated[observed-1,0,66:132]=0;generated[observed-1,0,258:262]=0
            positions=generated[...,:66].reshape(length,2,22,3)
            dest=out/arm;dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f'sample_{index:03d}.npz',features=generated,positions=positions,fps=30)
            score_start=cutoff if history_only else 0
            row['arms'][arm]={'defender_position_rmse_m':float(np.sqrt(np.mean((positions[score_start:,1]-truth_positions[score_start:,1])**2))),
                'known_actor_max_error_m':float(np.abs(positions[:observed,0]-truth_positions[:observed,0]).max()),
                'contact':evaluate_event(positions,radii,30,event)}
            if arm=='actor_conditioned' and not history_only:
                refined,history=refine_contacts(positions,radii,30,[event],edges,steps=80,device='cuda',fixed_actor=0)
                parent,child=np.asarray(edges).T
                lengths_before=np.linalg.norm(positions[:,:,parent]-positions[:,:,child],axis=-1)
                lengths_after=np.linalg.norm(refined[:,:,parent]-refined[:,:,child],axis=-1)
                bone_change=np.abs(lengths_after-lengths_before)
                dest=out/'actor_conditioned_contact';dest.mkdir(exist_ok=True)
                # Position-only refinement does not invent compatible rotations.
                np.savez_compressed(dest/f'sample_{index:03d}.npz',positions=refined,fps=30)
                row['arms']['actor_conditioned_contact']={'defender_position_rmse_m':float(np.sqrt(np.mean((refined[:,1]-truth_positions[:,1])**2))),
                    'known_actor_max_error_m':float(np.abs(refined[:,0]-truth_positions[:,0]).max()),
                    'mean_displacement_m':float(np.linalg.norm(refined-positions,axis=-1).mean()),
                    'max_bone_length_change_m':float(bone_change.max()),
                    'p95_bone_length_change_m':float(np.percentile(bone_change,95)),
                    'max_defender_joint_speed_m_s':float(np.linalg.norm(np.diff(refined[:,1],axis=0)*30,axis=-1).max()),
                    'contact':evaluate_event(refined,radii,30,event),'final_optimization':history[-1]}
        results.append(row);print('REACTION_CASE '+json.dumps(row),flush=True)
    (out/'evaluation.json').write_text(json.dumps({'cases':results,'adapter_sha256':hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        'test_source_disjoint':True,'selection':'first sorted test pseudo-event sources, independent of output',
        'conditions':'observed actor prefix plus caption and supplied pseudo-event contact type/time' if history_only else 'complete actor trajectory plus caption and supplied pseudo-event contact type/time',
        'history_only':history_only,'observed_frames':'min(30, length//2)' if history_only else 'all',
        'actor_future_as_input':not history_only,
        'online_causal':False,'contact_ground_truth':'unreviewed geometry, not measured forces',
        'physics_verified':False,'utility_verified':False},indent=2))
    print('ACTOR_REACTION_EVALUATION_COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
