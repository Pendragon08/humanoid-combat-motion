"""Call upstream train_agent unchanged; observe step counts and checkpoint writes."""
import argparse
import hashlib
import json
import random
import runpy
import sys
import time
from pathlib import Path
from isaacgym import gymapi
import numpy as np
import torch
from humanoidverse.agents.mh_ppo.mh_ppo import MHPPO

p=argparse.ArgumentParser();p.add_argument('--receipt-dir',type=Path,required=True)
p.add_argument('--expected-updates',type=int,required=True);p.add_argument('--expected-envs',type=int,required=True)
p.add_argument('--seed',type=int,default=1);p.add_argument('overrides',nargs=argparse.REMAINDER)
a=p.parse_args();a.receipt_dir=a.receipt_dir.resolve();assert not a.receipt_dir.exists()
a.receipt_dir.mkdir(parents=True)
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
counts=dict(environment_steps=0,updates=0,learn_reset_environment_steps=0,started=time.time());last={}
old_learn=MHPPO.learn;old_save=MHPPO.save;old_rollout=MHPPO._rollout_step
inside_rollout=[False]

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def observed_learn(self):
    assert self.num_learning_iterations==a.expected_updates and self.env.num_envs==a.expected_envs
    assert self.num_steps_per_env==24
    step=self.env.step
    def observed_step(*args,**kwargs):
        value=step(*args,**kwargs)
        if inside_rollout[0]:
            counts['environment_steps']+=self.env.num_envs
            counts['updates']=counts['environment_steps']//(self.env.num_envs*24)
        else:counts['learn_reset_environment_steps']+=self.env.num_envs
        return value
    self.env.step=observed_step
    return old_learn(self)

def observed_rollout(self,*args,**kwargs):
    inside_rollout[0]=True
    try:return old_rollout(self,*args,**kwargs)
    finally:inside_rollout[0]=False

def observed_save(self,path,infos=None):
    result=old_save(self,path,infos)
    record=dict(counts,checkpoint=str(Path(path).resolve()),upstream_iter_field=self.current_learning_iteration,
        seed=a.seed,explicit_launcher_seed=True,scope='Unmodified upstream algorithm with observation-only step/save wrappers',
        checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    write(Path(str(path)+'.receipt.json'),record);write(a.receipt_dir/'progress.json',record)
    last.update(record)
    return result

MHPPO.learn=observed_learn;MHPPO.save=observed_save;MHPPO._rollout_step=observed_rollout
overrides=a.overrides[1:] if a.overrides and a.overrides[0]=='--' else a.overrides
sys.argv=['humanoidverse/train_agent.py']+overrides
try:
    runpy.run_path('humanoidverse/train_agent.py',run_name='__main__')
    assert counts['environment_steps']==a.expected_updates*a.expected_envs*24
    assert last['environment_steps']==counts['environment_steps']
    write(a.receipt_dir/'TRAINING_COMPLETE.json',dict(last,finished=time.time()))
    print('PBHC_OFFICIAL_TRAINING_COMPLETE',json.dumps(last),flush=True)
except BaseException as exc:
    write(a.receipt_dir/'FAILED.json',dict(counts,error=repr(exc)));raise
