"""Train source-pinned body adaptations with recording-disjoint validation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from combat.remocap_data import digest
from combat.remocap_models import BodyModel, PINS, cosine_alphas, q_sample, sample_reaction


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['remos','intergen'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True)
    stat=np.load(a.data/'normalization.npz')
    mean=torch.tensor(stat['mean'],device='cuda')
    std=torch.tensor(stat['std'],device='cuda')
    arrays={s:np.load(a.data/f'{s}.npy') for s in ['train','val']}
    frames=arrays['train'].shape[1]
    model=BodyModel(a.method,frames=frames)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    alphas=cosine_alphas(300,'cuda')
    train_rng=np.random.default_rng(a.seed)
    val_ids=np.linspace(0,len(arrays['val'])-1,min(32,len(arrays['val'])),dtype=int)

    def batch_loss(raw,generator):
        pair=(torch.as_tensor(raw,device='cuda')-mean)/std
        actor=pair[:,:,0]
        target=pair[:,:,1] if a.method=='remos' else pair
        t=torch.randint(0,300,(len(pair),),generator=generator,device='cuda')
        noise=torch.randn(target.shape,device='cuda',generator=generator)
        pred=model(q_sample(target,t,noise,alphas),t,actor)
        loss=(pred-target).square().mean()
        reactor_loss=loss if a.method=='remos' else (pred[:,:,1]-target[:,:,1]).square().mean()
        return loss,reactor_loss

    @torch.no_grad()
    def validate():
        model.eval()
        gen=torch.Generator(device='cuda').manual_seed(1729)
        losses=[];sizes=[]
        for start in range(0,len(val_ids),a.batch_size):
            ids=val_ids[start:start+a.batch_size]
            _,loss=batch_loss(arrays['val'][ids],gen)
            losses.append(loss.item());sizes.append(len(ids))
        model.train()
        return float(np.average(losses,weights=sizes))

    provenance=dict(method=a.method+'-body-adapted',upstream_pins=PINS,seed=a.seed,
        steps=a.steps,batch_size=a.batch_size,frames=frames,learning_rate=1e-4,
        pilot=a.pilot,official_paper_reproduction=False,
        train_sha256=digest(a.data/'train.npy'),val_sha256=digest(a.data/'val.npy'),
        normalization_sha256=digest(a.data/'normalization.npz'),
        test_files_read=False,conditions='actor only at generation; no captions or contact labels',
        training_objective='normalized position diffusion MSE; no published auxiliary losses',
        validation_selection='32 deterministic source-disjoint val windows; reactor denoising MSE',
        parameters=sum(p.numel() for p in model.parameters()),
        code_sha256={str(Path(f).relative_to(Path(__file__).resolve().parents[1])):digest(f)
            for f in [__file__,Path(__file__).resolve().parents[1]/'combat/remocap_models.py']})
    (a.output/'config.json').write_text(json.dumps(provenance,indent=2))
    initial=validate();best=initial;best_step=0
    torch.save({'model':model.state_dict(),'provenance':provenance,'step':0},a.output/'best.pt')
    begin=time.monotonic(); generator=torch.Generator(device='cuda').manual_seed(a.seed+891)
    torch.cuda.reset_peak_memory_stats()
    with (a.output/'loss.jsonl').open('w') as log:
        for step in range(1,a.steps+1):
            ids=train_rng.integers(0,len(arrays['train']),size=a.batch_size)
            opt.zero_grad(set_to_none=True)
            loss,_=batch_loss(arrays['train'][ids],generator)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradients')
            opt.step()
            row=dict(step=step,loss=loss.item(),elapsed_seconds=time.monotonic()-begin)
            if step%100==0 or step==a.steps:
                score=validate();row['validation_reactor_mse']=score
                if score<best:
                    best=score;best_step=step
                    torch.save({'model':model.state_dict(),'provenance':provenance,'step':step},a.output/'best.pt')
            log.write(json.dumps(row)+'\n');log.flush()
            if step%25==0: print(json.dumps(row),flush=True)
    checkpoint=torch.load(a.output/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval()
    # Generation sanity check on validation only, never on test during fitting.
    raw=torch.tensor(arrays['val'][val_ids[:4]],device='cuda')
    actor=(raw[:,:,0]-mean)/std
    pred=sample_reaction(model,actor,alphas,steps=50,seed=20260916)*std+mean
    if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite sampled validation motion')
    mpjpe=torch.linalg.vector_norm(pred-raw[:,:,1],dim=-1).mean().item()*1000
    generated_bones=torch.linalg.vector_norm(pred[:,:,1:]-pred[:,:,np.array([0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])],dim=-1)
    report=dict(method=provenance['method'],seed=a.seed,steps=a.steps,pilot=a.pilot,
        initial_val_reactor_mse=initial,best_val_reactor_mse=best,best_step=best_step,
        validation_generation_mpjpe_mm=mpjpe,validation_generation_finite=True,
        generated_bone_median_m=generated_bones.median().item(),
        elapsed_seconds=time.monotonic()-begin,peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        best_checkpoint_sha256=digest(a.output/'best.pt'),
        training_pass=bool(best_step>0 and best<initial),
        test_evaluated=False,robot_utility_verified=False)
    (a.output/'TRAINING_COMPLETE.json').write_text(json.dumps(report,indent=2))
    print('REMOCAP_TRAINING_COMPLETE '+json.dumps(report),flush=True)


if __name__=='__main__':main()
