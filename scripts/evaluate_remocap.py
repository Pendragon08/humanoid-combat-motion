"""Frozen-checkpoint evaluation; full raw test plus fixed source-balanced ablation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from combat.remocap_data import digest
from combat.remocap_models import BodyModel, cosine_alphas, sample_reaction
from combat.remocap_eval import TrainContactLibrary,RADII,metrics
from combat.data_refinement import refine_data


def aggregate(rows):
    numeric=['mpjpe_mm','velocity_error_mm_s','acceleration_rms_m_s2','bone_length_temporal_std_mm',
             'bone_length_error_mm','proximity_gap_mae_mm','endpoint_sphere_penetration_fraction']
    output={}
    for track in ['full_test_raw','source_balanced_ablation']:
        output[track]={}
        for arm in sorted({r['arm'] for r in rows if r['track']==track}):
            selected=[r for r in rows if r['track']==track and r['arm']==arm]
            groups={s:[r for r in selected if r['source']==s] for s in {r['source'] for r in selected}}
            means={s:{k:float(np.mean([r['metrics'][k] for r in group])) for k in numeric} for s,group in groups.items()}
            total={key:sum(r['metrics'][key] for r in selected) for key in ['proximity_true_positive','proximity_false_positive','proximity_false_negative']}
            den=2*total['proximity_true_positive']+total['proximity_false_positive']+total['proximity_false_negative']
            output[track][arm]=dict(source_count=len(groups),samples=len(selected),
                source_macro_mean={k:float(np.mean([v[k] for v in means.values()])) for k in numeric},
                proximity_micro_f1_proxy=2*total['proximity_true_positive']/den if den else None,
                source_means=means)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--validation-smoke',action='store_true',help='Four validation windows only; never opens test motion')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4)
    checkpoint=torch.load(a.run/'best.pt',map_location='cpu',weights_only=False)
    prov=checkpoint['provenance']; method=prov['method'].split('-')[0]
    if not a.validation_smoke and (prov['pilot'] or prov['steps']!=5000):raise ValueError('Final evaluation requires declared main training')
    for name in ['train','val']:
        if digest(a.data/f'{name}.npy')!=prov[f'{name}_sha256']:raise ValueError('Training data changed')
    if digest(a.data/'normalization.npz')!=prov['normalization_sha256']:raise ValueError('Normalization changed')
    model=BodyModel(method,frames=prov['frames']);model.load_state_dict(checkpoint['model']);model.eval()
    del checkpoint
    split='val' if a.validation_smoke else 'test'
    test=np.load(a.data/f'{split}.npy')
    metadata=json.loads((a.data/f'{split}_windows.json').read_text())
    if a.validation_smoke:test=test[:4];metadata=metadata[:4]
    training_meta=json.loads((a.data/'train_windows.json').read_text())
    val_meta=json.loads((a.data/'val_windows.json').read_text())
    train_sources={r['source'] for r in training_meta};val_sources={r['source'] for r in val_meta}
    forbidden=train_sources if a.validation_smoke else train_sources|val_sources
    if {r['source'] for r in metadata}&forbidden:raise ValueError('Split leakage')
    library=TrainContactLibrary(np.load(a.data/'train.npy'),training_meta)
    # First chronological window per source, chosen without inspecting outputs or targets.
    chosen={}
    for i,r in enumerate(metadata):chosen.setdefault(r['source'],i)
    subset=set(chosen.values())
    stats=np.load(a.data/'normalization.npz')
    mean=torch.tensor(stats['mean'],device='cuda');std=torch.tensor(stats['std'],device='cuda')
    alphas=cosine_alphas(300,'cuda');a.output.mkdir(parents=True)
    protocol=dict(training=prov,checkpoint_sha256=digest(a.run/'best.pt'),evaluation_data_sha256=digest(a.data/f'{split}.npy'),
        split=split,validation_smoke=a.validation_smoke,test_files_read=not a.validation_smoke,
        conditions='full actor only',test_targets_used_for_generation=False,training_library_sources=sorted(train_sources),
        ablation_window_indices=sorted(subset),ablation_contact='train-only nearest-actor geometric touch proposal; not verified hit/block',
        temporal_weight=1e-5,refinement_steps=150,contact_weight=1.,
        generation_seed=20260916,draws=3,ddim_steps=50,
        contact_metric='endpoint spheres; geometric proxy, not independent semantic/force labels',
        official_fid_computed=False,official_paper_reproduction=False)
    root=Path(__file__).resolve().parents[1]
    protocol['code_sha256']={name:digest(root/name) for name in [
        'scripts/evaluate_remocap.py','combat/remocap_eval.py','combat/remocap_models.py',
        'combat/data_refinement.py','combat/contact_guidance.py','combat/interaction.py']}
    protocol['training_library_sha256']=digest(a.data/'train.npy')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];begin=time.monotonic()
    # Saved locally only: license prohibits redistributing derived motion.
    generated=np.empty((3,)+test[:,:,1].shape,dtype=np.float32)
    with (a.output/'metrics.jsonl').open('w') as log:
        for draw in range(3):
            for start in range(0,len(test),a.batch_size):
                batch=test[start:start+a.batch_size]
                actor=(torch.tensor(batch[:,:,0],device='cuda')-mean)/std
                pred=(sample_reaction(model,actor,alphas,50,20260916+draw*100000+start)*std+mean).cpu().numpy()
                if not np.isfinite(pred).all():raise FloatingPointError('Invalid generation')
                generated[draw,start:start+len(batch)]=pred
                for offset,x in enumerate(pred):
                    i=start+offset;given=test[i,:,0].copy();truth=test[i,:,1]
                    base=dict(window=i,source=metadata[i]['source'],draw=draw)
                    def record(arm,y,track,extra=None):
                        row=dict(**base,track=track,arm=arm,metrics=metrics(given,y,truth))
                        if extra:row.update(extra)
                        rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    record('raw',x,'full_test_raw')
                    if i in subset:
                        event,info=library.propose(given)  # no held-out reactor argument
                        pair=np.stack([given,x],axis=1)
                        record('raw',x,'source_balanced_ablation')
                        for arm,cw,aw in [('temporal',0.,1e-5),('contact',1.,0.),('contact_temporal',1.,1e-5)]:
                            if event is None and aw==0:
                                y=pair.copy()
                            else:
                                y,_=refine_data(pair,RADII,20,event,cw if event else 0.,aw,steps=150,device='cuda')
                            if not np.allclose(y[:,0],given,atol=1e-7):raise ValueError('Condition actor moved')
                            record(arm,y[:,1],'source_balanced_ablation',dict(contact_plan=info,contact_applied=bool(event and cw)))
                if start%40==0:print(json.dumps(dict(draw=draw,windows_done=start+len(batch),total_windows=len(test),elapsed_seconds=time.monotonic()-begin)),flush=True)
    np.save(a.output/'generated_reactors.npy',generated)
    diversity=np.mean([np.linalg.norm(generated[i]-generated[j],axis=-1).mean() for i,j in [(0,1),(0,2),(1,2)]])
    result=dict(method=prov['method'],training_seed=prov['seed'],aggregate=aggregate(rows),
        validation_smoke=a.validation_smoke,split=split,
        within_condition_diversity_mm=float(diversity*1000),elapsed_seconds=time.monotonic()-begin,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        independent_contact_ground_truth=False,robot_utility_verified=False,
        checkpoint_sha256=protocol['checkpoint_sha256'],metrics_sha256=digest(a.output/'metrics.jsonl'),
        generated_motion_sha256=digest(a.output/'generated_reactors.npy'))
    (a.output/'EVALUATION_COMPLETE.json').write_text(json.dumps(result,indent=2))
    print('REMOCAP_EVALUATION_COMPLETE '+json.dumps({k:v for k,v in result.items() if k!='aggregate'}),flush=True)


if __name__=='__main__':main()
