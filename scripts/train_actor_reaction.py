"""Train an offline actor-conditioned adapter on explicitly unreviewed pseudo roles.

Full actor trajectory and supplied contact window are conditions. This is not an
online causal reaction policy, verified combat-role annotation, or physics model.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from combat.annotations import read_annotation
from combat.protocol import load_manifest
from combat.checkpoint_io import load_model_state
from combat.reaction import ActorMotionAdapter,CausalReactionResidual,masked_defender_mse,history_condition
from scripts.train_timotion_adapter import fixed_validation_rng


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--frames',type=int,default=120);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--history-only',action='store_true',help='Observe actor prefix only; predict future defender')
    p.add_argument('--cross-attention',action='store_true')
    a=p.parse_args();out=a.output.resolve();manifest=a.manifest.resolve();events_path=a.events.resolve()
    if out.exists():raise FileExistsError(out)
    records={r['sample_id']:r for r in load_manifest(manifest)}
    events=[json.loads(line) for line in events_path.read_text().splitlines() if line]
    root=Path(__file__).resolve().parents[1];import clip
    clip.load=functools.partial(clip.load,download_root=str(root/'cache/clip'))
    os.chdir(root/'third_party/TIMotion')
    from configs import get_config
    from models import TIMotion
    from utils.preprocess import load_motion
    from utils.utils import process_motion_np,qmul_np,qinv_np,qrot_np,rigid_transform,trans_matrix
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    cfg=get_config('configs/model.yaml');cfg.NUM_HEADS=16;cfg.NUM_LAYERS=5;cfg.LATENT_DIM=512;cfg.LPA=False
    model=TIMotion(cfg);weights,sha=load_model_state(root/'data/official/timotion_weights.npz')
    model.load_state_dict(weights,strict=True);del weights;model=model.cuda().eval().requires_grad_(False)
    if a.cross_attention and not a.history_only:raise ValueError('Cross attention experiment requires explicit history-only protocol')
    adapter=ActorMotionAdapter().cuda();reaction=CausalReactionResidual().cuda() if a.cross_attention else None
    parameters=list(adapter.parameters())+(list(reaction.parameters()) if reaction else [])
    opt=torch.optim.AdamW(parameters,lr=1e-4)
    splits={'train':[],'val':[]}
    for item in events:
        if item['split'] not in splits:continue  # Do not read test motion files.
        record=records[item['sample_id']]
        if record['split']!=item['split']:raise ValueError('Split mismatch')
        order=[item['event']['a_actor'],item['event']['b_actor']]
        arrays=[load_motion(record['motion_paths'][i],16,False)[0] for i in order]
        center=(item['event']['start']+item['event']['end'])//2
        start=max(0,min(center-a.frames//2,len(arrays[0])-a.frames-1))
        arrays=[x[start:start+a.frames+1].copy() for x in arrays]
        first,qa,pa=process_motion_np(arrays[0],.001,0,22)
        second,qb,pb=process_motion_np(arrays[1],.001,0,22)
        relative=qmul_np(qb,qinv_np(qa));angle=np.arctan2(relative[:,2:3],relative[:,0:1]);xz=qrot_np(qa,pb-pa)[:,[0,2]]
        second=rigid_transform(np.concatenate([angle,xz],axis=-1)[0],second)
        pair=np.concatenate([first,second],axis=-1);length=len(pair)
        if length<15:continue
        cutoff=min(30,length//2)
        if a.history_only:
            transformed=[np.einsum('mn,tjn->tjm',trans_matrix,x[:,:66].reshape(-1,22,3)) for x in arrays]
            shared_floor=min(x[:cutoff,:,1].min() for x in transformed)
            for actor,x in enumerate(transformed):
                # Undo future-derived per-person floor; common observed floor.
                pair[:,actor*262+1:actor*262+66:3]+=x[:,:,1].min()-shared_floor
                # Recompute foot bits after the floor change, from forward speed.
                for k,j in enumerate((7,10,8,11)):
                    pos=pair[:,actor*262+j*3:actor*262+j*3+3]
                    vel=pair[:,actor*262+66+j*3:actor*262+66+j*3+3]
                    pair[:,actor*262+258+k]=((vel**2).sum(-1)<.001)&(pos[:,1]<(0.12 if k%2==0 else .05))
        texts=[s for s in read_annotation(record['annotation_path'])[0].splitlines() if s.strip()]
        window=[max(0,item['event']['start']-start)/length,min(length-1,item['event']['end']-start)/length]
        splits[item['split']].append({'id':item['sample_id'],'motion':np.pad(pair,((0,a.frames-length),(0,0))),
            'length':length,'texts':texts,'kind':1 if item['event']['kind']=='hit' else 2,'window':window,'start':start,'cutoff':cutoff})
    if len(splits['train'])<20 or len(splits['val'])<5:raise ValueError('Insufficient source-disjoint pseudo-event data')

    def batch(rows,rng):
        raw=torch.tensor(np.stack([r['motion'] for r in rows]),dtype=torch.float32,device='cuda')
        lengths=torch.tensor([r['length'] for r in rows],device='cuda')
        normalized=model.decoder.diffusion.normalizer.forward(raw.reshape(len(rows),a.frames,2,262)).reshape(len(rows),a.frames,524)
        with torch.no_grad():text=model.text_process({'text':[rng.choice(r['texts']) for r in rows]})['cond']
        cutoffs=torch.tensor([r['cutoff'] for r in rows],device='cuda')
        actor=history_condition(raw[...,:262],cutoffs,model.decoder.diffusion.normalizer) if a.history_only else normalized[...,:262]
        cond=adapter(text,actor,cutoffs if a.history_only else lengths,torch.tensor([r['kind'] for r in rows],device='cuda'),
                     torch.tensor([r['window'] for r in rows],device='cuda'))
        mask=(torch.arange(a.frames,device='cuda')[None]<lengths[:,None]).float()[:,:,None].expand(-1,-1,2)
        times,_=model.decoder.sampler.sample(len(rows),'cuda')
        noisy=model.decoder.diffusion.q_sample(normalized,times,noise=torch.randn_like(normalized))
        if a.history_only:
            for i,cutoff in enumerate(cutoffs.tolist()):
                noisy[i,cutoff:,:262]=torch.randn_like(noisy[i,cutoff:,:262])
                # Last observed frame's forward-difference channels must not leak.
                noisy[i,cutoff-1,66:132]=0;noisy[i,cutoff-1,258:262]=0
        prediction=model.decoder.net(noisy,times,mask=mask,cond=cond)
        if reaction is not None:prediction=reaction(prediction,actor,cutoffs)
        return masked_defender_mse(prediction,normalized,lengths,cutoffs if a.history_only else None)

    def validate():
        values=[];sizes=[];adapter.eval()
        if reaction is not None:reaction.eval()
        with fixed_validation_rng():
            rng=random.Random(1729)
            with torch.no_grad():
                for k in range(0,len(splits['val']),a.batch_size):
                    rows=splits['val'][k:k+a.batch_size];values.append(float(batch(rows,rng)));sizes.append(len(rows))
        return float(np.average(values,weights=sizes))
    out.mkdir(parents=True);baseline=validate();best=float('inf');rng=random.Random(a.seed);begin=time.monotonic()
    provenance={'seed':a.seed,'steps':a.steps,'frames':a.frames,'batch_size':a.batch_size,
        'source_ids':{k:[r['id'] for r in v] for k,v in splits.items()},'backbone_sha256':sha,
        'event_file_sha256':hashlib.sha256(events_path.read_bytes()).hexdigest(),
        'model':'frozen TIMotion + trainable full-actor-motion/contact-window residual encoder',
        'labels':'unreviewed geometric pseudo-role and hit/block candidates',
        'loss':'masked normalized defender-only reconstruction MSE',
        'conditions':'full actor trajectory, caption, supplied event kind and time window',
        'online_causal':False,'dynamics_aware':False,'test_files_read':False,
        'zero_adapter_validation_loss':baseline,'trainable_parameters':sum(p.numel() for p in adapter.parameters())}
    if a.history_only:
        provenance.update(model='frozen TIMotion + observed actor-prefix adapter',
            conditions='up to 30 observed actor frames, caption, externally supplied event kind/window',
            actor_future_as_input=False,history_only=True,online_causal=False,
            floor_transform='common minimum over observed histories only',
            loss='future-defender normalized diffusion MSE; future actor input replaced by independent noise',
            caveat='prefix-conditioned offline forecast, not verified tactical causality or real-time deployment')
    provenance['cross_attention']=bool(a.cross_attention)
    provenance['trainable_parameters']=sum(p.numel() for p in parameters)
    (out/'config.json').write_text(json.dumps(provenance,indent=2))
    with (out/'loss.jsonl').open('w') as log:
        for step in range(1,a.steps+1):
            adapter.train()
            if reaction is not None:reaction.train()
            opt.zero_grad();loss=batch(rng.choices(splits['train'],k=a.batch_size),rng)
            if not torch.isfinite(loss):raise FloatingPointError('Invalid reaction loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,.5);opt.step()
            record={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-begin}
            if step%100==0 or step==a.steps:
                score=validate();record['validation_loss']=score
                if score<best:
                    best=score;torch.save({'adapter':adapter.state_dict(),'reaction':reaction.state_dict() if reaction is not None else None,'provenance':provenance,'step':step},out/'best_adapter.pt')
            log.write(json.dumps(record)+'\n');log.flush()
            if step%50==0:print(json.dumps(record),flush=True)
    result={'steps':a.steps,'baseline_validation_loss':baseline,'best_validation_loss':best,
            'independent_test':False,'causal_correctness_verified':False,'robot_utility_verified':False}
    (out/'TRAINING_COMPLETE.json').write_text(json.dumps(result,indent=2));print('ACTOR_REACTION_TRAINING_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':main()
