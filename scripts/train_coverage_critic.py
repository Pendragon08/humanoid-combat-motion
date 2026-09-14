"""Retrospective 48-rollout audit with prompt-group separation and abstention.

All targets describe ONE fixed controller. Never fills unobserved contact/fall/
collision labels. Three predictor seeds are not three controllers or experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from combat.coverage_critic import ControllerCoverageCritic,reference_features,prompt_split

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
root=Path(__file__).resolve().parents[1];rows=[];features=[];controllers=set();prompts=[]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arm in ('generic','adapted'):
    directory=root/f'runs/timotion_{arm}_24_seed2026'
    gen=json.loads((directory/'manifest.json').read_text())
    for item in gen['samples']:
        name=item['sample_id'];text=item['text']
        if text not in prompts:prompts.append(text)
        group=prompts.index(text);split=prompt_split(group)
        metrics_path=root/f'runs/dual_g1_{arm}_{name}_iter1098/metrics.json'
        metrics=json.loads(metrics_path.read_text());controllers.add(metrics['checkpoint_sha256'])
        rollout_path=metrics_path.parent/'rollout.npz'
        with np.load(rollout_path,allow_pickle=False) as z:fps=float(z['fps'])
        if metrics['robots_per_environment']!=2 or not metrics['learned_tracking']:raise ValueError('Wrong controller provenance')
        if max(metrics['first_failure_steps'])>=metrics['horizon_steps']:raise ValueError('Censored survivors need survival modeling, not this failure-time regression')
        refs=root/f'runs/timotion_{arm}_24_seed2026_tracking_refs'
        for actor in range(2):
            if digest(refs/f'{name}_actor{actor}.npz')!=metrics['reference_sha256'][actor]:raise ValueError('Reference changed after rollout')
        qpath=root/f'runs/timotion_{arm}_24_seed2026_g1ik/{name}.npz'
        ik_manifest=json.loads((qpath.parent/'manifest.json').read_text())
        expected=next(r['sha256'] for r in ik_manifest['samples'] if r['sample']==name)
        if digest(qpath)!=expected:raise ValueError('IK checksum mismatch')
        with np.load(qpath,allow_pickle=False) as z:x=reference_features(z['qpos'],float(z['fps']))
        features.append(x)
        rows.append({'id':arm+'_'+name,'arm':arm,'prompt_group':group,'split':split,
            'target_first_tracking_failure_s':float(np.mean(metrics['first_failure_steps'])/fps),
            'completion_fraction':metrics['completion_fraction'],'input_sha256':digest(qpath),
            'metrics_sha256':digest(metrics_path),'rollout_sha256':digest(rollout_path),
            'full_physics_labels_approved':metrics.get('critic_training_labels_approved',False)})
if len(controllers)!=1 or len(rows)!=48 or len(prompts)!=4:raise ValueError('Unexpected source/controller inventory')
indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','val','test')}
train_x=np.concatenate([features[i] for i in indices['train']])
mean=train_x.mean(0);std=np.maximum(train_x.std(0),1e-4)
y=np.array([r['target_first_tracking_failure_s'] for r in rows],np.float32)
ym=float(y[indices['train']].mean());ys=max(float(y[indices['train']].std()),.01)
def batch(ids):
    return pad_sequence([torch.tensor((features[i]-mean)/std) for i in ids],batch_first=True),torch.tensor([len(features[i]) for i in ids])
xtr,ltr=batch(indices['train']);xv,lv=batch(indices['val']);xt,lt=batch(indices['test'])
a.output.mkdir(parents=True);seed_results=[]
for seed in (0,1,2):
    torch.manual_seed(seed);np.random.seed(seed)
    model=ControllerCoverageCritic();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    target=torch.tensor((y[indices['train']]-ym)/ys)
    history=[];best=float('inf');best_state=None
    for epoch in range(150):
        model.train();opt.zero_grad();loss=(model(xtr,ltr)-target).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();model.eval()
        with torch.no_grad():val_pred=model(xv,lv).numpy()*ys+ym
        mae=float(np.abs(val_pred-y[indices['val']]).mean())
        history.append({'epoch':epoch,'train_loss':float(loss),'validation_mae_s':mae})
        if mae<best:best=mae;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():pred=model(xt,lt).numpy()*ys+ym
    target_test=y[indices['test']];rankings=[]
    for arm in ('generic','adapted'):
        ids=[j for j,i in enumerate(indices['test']) if rows[i]['arm']==arm]
        chosen=ids[int(np.argmax(pred[ids]))]
        rankings.append({'arm':arm,'K':len(ids),'selected_id':rows[indices['test'][chosen]]['id'],
            'selected_actual_failure_s':float(target_test[chosen]),'uniform_random_expected_failure_s':float(target_test[ids].mean()),
            'oracle_actual_failure_s':float(target_test[ids].max()),
            'selected_completion_fraction':rows[indices['test'][chosen]]['completion_fraction']})
    record={'seed':seed,'best_epoch':best_epoch,'validation_mae_s':best,
            'validation_constant_baseline_mae_s':float(np.abs(y[indices['val']]-ym).mean()),
            'test_mae_s':float(np.abs(pred-target_test).mean()),
            'test_constant_baseline_mae_s':float(np.abs(target_test-ym).mean()),
            'test_predictions':[dict(rows[i],prediction_failure_s=float(pred[j])) for j,i in enumerate(indices['test'])],
            'retrospective_reranking':rankings}
    seed_results.append(record)
    torch.save({'state_dict':best_state,'mean':torch.tensor(mean),'std':torch.tensor(std),'target_mean':ym,'target_std':ys,
                'target':'conditional tracking failure time only','controller_sha256':next(iter(controllers)),
                'physics_feasibility_critic':False,'approved_for_candidate_deployment':False},a.output/f'seed{seed}.pt')
    (a.output/f'history_seed{seed}.json').write_text(json.dumps(history,indent=2))
    print('COVERAGE_CRITIC_SEED',json.dumps(record),flush=True)
report={'experiment':'retrospective controller-coverage predictor; not the full four-head physics critic',
        'records':rows,'split_counts':{k:len(v) for k,v in indices.items()},'prompt_groups':prompts,
        'split_rule':'first two prompt groups train; third validation; fourth test; all latent repeats and arms stay together',
        'controller_sha256':next(iter(controllers)),'feature_input':'reference qpos and finite-difference reference root/joint velocity; no realized state or errors',
        'results':seed_results,'full_physics_critic_trained':False,'contact_labels_imputed':False,
        'candidate_deployment_approved':False,'full_horizon_success_records':sum(r['completion_fraction']>0 for r in rows),
        'limitations':['all original rollouts already inspected; retrospective diagnostic',
            'only four prompt groups; one held-out group; four nominal environments are not independent samples',
            'one underperforming fixed controller; does not estimate intrinsic robot feasibility',
            'ranking a few hundred milliseconds of failure does not produce usable references',
            'no observed full-horizon successes; contact/fall/full collision unobserved or censored'],
        'deployment_decision':'ABSTAIN: no full-horizon successful calibration examples and no complete physics labels'}
(a.output/'report.json').write_text(json.dumps(report,indent=2))
print('COVERAGE_CRITIC_COMPLETE',flush=True)
