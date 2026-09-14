"""Train a critic ONLY from verified paired learned-tracker rollout records.

Inputs: reference_qpos/reference_qvel in NPZ, not realized rollout states or
tracking errors. Targets: actual physics outcomes, kept in separate records.
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from combat.paired_models import DualRobotPhysicsCritic,critic_loss
from combat.protocol import load_manifest,validate_physics_label,check_artifact


def reference_features(path):
    with np.load(path,allow_pickle=False) as z:
        q,v=z['reference_qpos'],z['reference_qvel']
    if q.ndim!=3 or v.ndim!=3 or q.shape[:2]!=v.shape[:2] or q.shape[1]!=2 or len(q)<2:
        raise ValueError('References must be [T>=2,2,DOF]')
    x=np.concatenate([q,v],axis=-1).reshape(len(q),-1).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite reference')
    return x


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False
    records=load_manifest(args.manifest)
    # Test data are not even read by this training entry point.
    records=[r for r in records if r['split'] in ('train','val')]
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
    if len(train)<20 or len(val)<10:
        raise ValueError('Need >=20 training and >=10 validation source records; tiny toy labels are not a critic experiment')
    features={}
    for r in records:
        validate_physics_label(r)
        for path_key,hash_key in [('reference_path','reference_sha256'),('rollout_path','rollout_sha256'),('controller_checkpoint_path','controller_checkpoint_sha256')]:
            check_artifact(r[path_key],r[hash_key])
        features[r['sample_id']]=reference_features(r['reference_path'])
    dims={x.shape[1] for x in features.values()}
    if len(dims)!=1:raise ValueError('Mixed robot feature dimensions require explicit morphology encoding')
    train_frames=np.concatenate([features[r['sample_id']] for r in train])
    mean=train_frames.mean(0);std=np.maximum(train_frames.std(0),1e-4)
    model=DualRobotPhysicsCritic(len(mean)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    target_names=['tracking_error_m','fall','contact_success','invalid_collision']
    def batch(rows):
        seq=[torch.tensor((features[r['sample_id']]-mean)/std,device='cuda') for r in rows]
        lengths=torch.tensor([len(s) for s in seq],device='cuda')
        targets={k:torch.tensor([r[k] for r in rows],device='cuda',dtype=torch.float32) for k in target_names}
        return pad_sequence(seq,batch_first=True),lengths,targets
    args.output.mkdir(parents=True)
    best=float('inf');history=[]
    for epoch in range(args.epochs):
        random.shuffle(train);model.train();total=0
        for start in range(0,len(train),args.batch_size):
            rows=train[start:start+args.batch_size];x,lengths,y=batch(rows)
            opt.zero_grad();loss=critic_loss(model(x,lengths),y)
            if not torch.isfinite(loss):raise FloatingPointError('Critic training diverged')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            total+=loss.item()*len(rows)
        model.eval();validation=0
        with torch.no_grad():
            for start in range(0,len(val),args.batch_size):
                rows=val[start:start+args.batch_size];x,lengths,y=batch(rows)
                validation+=critic_loss(model(x,lengths),y).item()*len(rows)
        validation/=len(val)
        record={'epoch':epoch,'train_loss':total/len(train),'validation_loss':validation}
        history.append(record);print(json.dumps(record),flush=True)
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        if validation<best:
            best=validation
            torch.save({'state_dict':model.state_dict(),'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
                        'seed':args.seed,'epoch':epoch,'input_dim':len(mean),'targets':target_names,
                        'train_sources':[r['source_ids'] for r in train],'validation_sources':[r['source_ids'] for r in val],
                        'validation_loss':validation,'trained_on':'paired_learned_tracker_rollouts'},args.output/'best.pt')
    (args.output/'status.json').write_text(json.dumps({'training_complete':True,'test_evaluation_complete':False,
                                                     'synthetic_data_utility_verified':False,'best_validation_loss':best},indent=2))


if __name__=='__main__':main()
