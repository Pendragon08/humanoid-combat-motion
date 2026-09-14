"""Freeze matched training-only augmentation arms; no test-conditioned outputs."""
import hashlib
import json
from pathlib import Path
from combat.reference_bundle import build_bundle

root=Path(__file__).resolve().parents[1]
out=root/'runs/step7_bundles'
if out.exists():raise FileExistsError(out)
read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
real=read('data/tracking_seed/kungfu_train20_bundle.json')['clips']
test=read('runs/kungfu_test6_qpos/source_manifest.json')['records']
report=read('runs/step7_train20_generation/evaluation.json')
if report.get('split')!='train' or not report.get('training_augmentation'):raise ValueError('Not training augmentation')
test_groups={r['source']['source_group'] for r in test}
if test_groups & {r['source_group'] for r in real}:raise ValueError('Video source leakage')
events=[json.loads(l) for l in (root/'records/paired_v1/pseudo_contact_events/events.jsonl').read_text().splitlines() if l]
train_sources={r['sample_id'] for r in events if r['split']=='train'}
sources=[c['source'] for c in report['cases']]
if len(sources)!=20 or len(set(sources))!=20 or not set(sources)<=train_sources:raise ValueError('Invalid augmentation sources')
out.mkdir()
for arm in ('generic','contact'):
    entries=[dict(r) for r in real]
    entries += [{'path':str(root/f'runs/step7_{arm}_tracking_refs/sample_{i:03d}_actor1.npz'),
        'clip_id':f'synthetic_{i:03d}', 'source_id':f'InterHuman:{source}',
        'source_group':f'InterHuman:{source}','split':'train','kind':'synthetic_human_retargeted',
        'conditioning_split':'train','actor':'defender','generator_arm':arm} for i,source in enumerate(sources)]
    build_bundle(entries,out/f'{arm}.npz')
protocol={'experiment':'Step 7 component-level synthetic-data utility pilot','seeds':[0,1,2],
 'training':{'iterations':1000,'num_envs':512,'steps_per_env':24,'env_steps_per_run':12288000,
             'initialization':'from scratch','selection':'final model_999; no selection on test','sampling':'uniform clip and valid frame'},
 'real':{'clips':20,'type':'video-reconstructed KungFuAthlete G1, not optical MoCap'},
 'generic':{'real_clips':20,'synthetic_clips':20,'method':'frozen TIMotion, complete attacker inpainting; defender reference'},
 'contact':{'real_clips':20,'synthetic_clips':20,'method':'actor adapter plus bone-preserving contact optimization; defender reference',
            'physics_critic_used':False,'dynamics_guaranteed':False},
 'matched_generation_seed':7100,'sources':sources,
 'test':{'clips':4,'independent_video_title_groups':len(test_groups),'groups':sorted(test_groups),
         'envs_per_clip':32,'reset_seed':123,'source_group_disjoint':True,
         'caveat':'Existing four clips are development holdouts already inspected before this protocol; not untouched confirmatory data.'},
 'seed0_real_reuse':{'run':'runs/multireference_kungfu20_seed0','reason':'identical 20-clip bundle, PPO settings, seed, and step budget; no new training'},
 'claims_allowed':['effect at this fixed training budget on these development holdouts'],
 'claims_not_allowed':['full contact/physics pipeline validated','paired combat utility','critic-selected data utility','broad population generalization'],
 'confounds':['generic vs contact compares adapter plus contact package, not contact-only ablation',
             'real vs augmented changes source distribution and per-clip exposure; fixed total optimization budget',
             'generator is trained once; seeds vary controller initialization, not generator dataset',
             'single-robot downstream policies do not retain physical attacker interaction']}
(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print('STEP7_BUNDLES_READY',json.dumps(protocol,ensure_ascii=False),flush=True)
