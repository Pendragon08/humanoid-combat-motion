"""Matched 2x2 post-generation ablation on saved training-only paired candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from combat.data_refinement import refine_data,quality_metrics
from combat.interaction import ContactEvent

ARMS={'raw':(0.,0.),'contact':(1.,0.),'temporal':(0.,1e-5),'contact_temporal':(1.,1e-5)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard-index',type=int,required=True)
    p.add_argument('--shard-count',type=int,default=2)
    p.add_argument('--steps',type=int,default=150)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if not 0<=a.shard_index<a.shard_count:raise ValueError('Invalid shard')
    if a.output.exists():raise FileExistsError(a.output)
    report=json.loads((a.input/'evaluation.json').read_text())
    if report['split']!='train' or not report['training_augmentation'] or report['history_only']:
        raise ValueError('Require full-attacker, training-only saved candidates')
    events={r['sample_id']:r for r in (json.loads(l) for l in a.events.read_text().splitlines() if l)}
    cases=report['cases']
    if len({c['source'] for c in cases})!=len(cases):raise ValueError('Repeated source')
    if any(events[c['source']]['split']!='train' for c in cases):raise ValueError('Nontraining condition')
    a.output.mkdir(parents=True)
    start=time.perf_counter(); rows=[]
    protocol={'experiment':'E2-component contact x temporal regularization',
              'scope':'post-generation human kinematic proxy ablation; not full physics-aware generator or downstream policy training',
              'status':'running','source_report_sha256':sha(a.input/'evaluation.json'),
              'events_sha256':sha(a.events),'arms':ARMS,'steps':a.steps,'lr':0.01,
              'shard_index':a.shard_index,'shard_count':a.shard_count,
              'input_arm':'inpainting_only','attacker_fixed':True,'preserve_source_bone_lengths':True,
              'conditions':'same saved full attacker, pseudo contact event, frames and original generation seed',
              'selection':'all 20 previous training conditions, no output-based rejection',
              'test_status':'retrospective training-set diagnostic; no untouched test evidence',
              'host':platform.node(),'torch':torch.__version__,'device':a.device,
              'gpu':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
              'module_sha256':sha(Path(__file__).resolve().parents[1]/'combat/data_refinement.py'),
              'runner_sha256':sha(Path(__file__)), 'new_policy_training_steps':0}
    save(a.output/'protocol.json',protocol)
    for index,case in enumerate(cases):
        if index%a.shard_count!=a.shard_index:continue
        path=a.input/'inpainting_only'/f'sample_{index:03d}.npz'
        with np.load(path,allow_pickle=False) as data:
            x=np.asarray(data['positions']); fps=float(data['fps'])
        radii=np.asarray(events[case['source']]['radii_m'])
        event=ContactEvent(**case['pseudo_event'])
        for arm,(cw,aw) in ARMS.items():
            t=time.perf_counter();torch.manual_seed(20260915+index)
            if arm=='raw': y=x.copy();history=[]
            else:y,history=refine_data(x,radii,fps,event,cw,aw,steps=a.steps,device=a.device)
            metrics=quality_metrics(y,x,radii,fps,event)
            destination=a.output/f'{index:03d}_{arm}.npz'
            np.savez_compressed(destination,positions=y,fps=fps)
            row={'index':index,'source':case['source'],'generation_seed':case['seed'],'arm':arm,
                 'source_sha256':sha(path),'output_sha256':sha(destination),'fps':fps,'frames':len(x),
                 'seconds':time.perf_counter()-t,'quality':metrics,'optimization':history}
            rows.append(row);save(a.output/'results.json',rows)
            print('DATA_ABLATION_CASE',json.dumps(row),flush=True)
    protocol.update(status='complete',cases=len(rows)//len(ARMS),outputs=len(rows),runtime_s=time.perf_counter()-start)
    save(a.output/'protocol.json',protocol)
    save(a.output/'COMPLETE.json',{'outputs':len(rows),'protocol_sha256':sha(a.output/'protocol.json')})
    print('DATA_ABLATION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
