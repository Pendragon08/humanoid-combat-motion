"""Finite upstream reproduction stages; full training requires independent witness."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_official_reproduction_20260917'
OUT=ROOT/'runs/pbhc_official_reproduction_20260917'
UP=ROOT/'third_party/PBHC'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value,(int,float)) else True

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','train'],required=True);a=p.parse_args()
    REC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    handle=(REC/'queue.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=REC/(a.stage+'_state.json');assert not statepath.exists(),'Preserve prior stage; no blind retry'
    cfg=json.loads((ROOT/'configs/pbhc_official_reproduction_v1.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()==cfg['commit']
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=UP,text=True).strip()
    state=dict(status='waiting_for_existing_queue',stage=a.stage,started=time.time());write(statepath,state)
    frozen={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'env-pbhc-training.json',ROOT/'configs/pbhc_official_reproduction_v1.json',
        ROOT/'scripts/run_pbhc_training.sh',ROOT/'scripts/observe_pbhc_training.py',ROOT/'scripts/witness_pbhc_gym.py',UP/cfg['motion'],UP/cfg['provided_checkpoint']]}
    def invoke(key,args):
        assert shutil.disk_usage(ROOT).free>20*1024**3
        for path,digest in frozen.items():assert sha(ROOT/path)==digest,path
        state.update(status='running',current_job=key);write(statepath,state)
        command=['bash',str(ROOT/'scripts/run_pbhc_training.sh')]+args
        with (REC/(key+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(REC/(key+'_exit.json'),dict(command=command,exit_code=result.returncode,finished=time.time()))
        if result.returncode:raise RuntimeError(key+' failed; inspect evidence, no automatic algorithm changes')
    def training(tag,envs,updates):
        folder=OUT/tag;assert not folder.exists()
        receipt=REC/(tag+'_receipt');assert not receipt.exists()
        args=[str(ROOT/'scripts/observe_pbhc_training.py'),'--receipt-dir',str(receipt),'--expected-updates',str(updates),
              '--expected-envs',str(envs),'--seed',str(cfg['training_seed']),'--']+cfg['training_overrides']+[
            'project_name=KungfuBotOfficialReproduction','experiment_name='+tag,'num_envs='+str(envs),
            'algo.config.num_learning_iterations='+str(updates),'robot.motion.motion_file='+cfg['motion'],
            'experiment_dir='+str(folder),'+device=cuda:0']
        invoke(tag,args)
        record=json.loads((receipt/'TRAINING_COMPLETE.json').read_text())
        assert record['environment_steps']==envs*updates*24
        checkpoint=Path(record['checkpoint']);assert checkpoint.exists() and sha(checkpoint)==record['checkpoint_sha256']
        return checkpoint
    def evaluate(tag,checkpoint,episodes=100):
        metricdir=checkpoint.parent/'metrics'/('ckpt_'+checkpoint.stem.split('_')[-1])
        outputs={}
        for mode,entry in [('accuracy','sample_eps.py'),('ratio','ratio_eps.py')]:
            label='official_'+tag+'_'+mode
            suffix='_ratio.json' if mode=='ratio' else '.json'
            target=metricdir/(label+suffix);assert not target.exists()
            invoke(label,['humanoidverse/'+entry,'+device=cuda:0','+checkpoint='+str(checkpoint),
                '+num_envs='+str(episodes),'+num_episodes='+str(episodes),'+eps_eval_name='+label,'+opt=record',
                'eval_log_dir='+str(OUT/'evaluation_logs'/label)])
            value=json.loads(target.read_text());assert finite(value)
            if mode=='ratio':assert 0<=value['ratio']<=1
            else:assert all(key in value['accuracy'] for key in ['E_gmpbpe','E_mpbpe','E_mpjpe','E_mpjve','E_pbve','E_pbae'])
            shutil.copy2(target,REC/(label+'.json'));outputs[mode]=dict(path=str(target),sha256=sha(target))
        return outputs
    try:
        if a.stage=='train':
            assert (REC/'PREFLIGHT_COMPLETE.json').exists()
            witness=json.loads((REC/'independent_witness.json').read_text())
            spec=json.loads((ROOT/'env-pbhc-training.json').read_text())
            spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
            assert witness['status']=='pass' and witness['spec_hash']==spec_hash
            assert witness['agent'] and witness['invocation_unmodified'] is True
        deadline=time.time()+6*3600
        while True:
            old=json.loads((ROOT/'records/pbhc_matched_data_20260917/state.json').read_text())
            memory=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
            if old['status'] in ['complete','failed'] and all(v<500 for v in memory):break
            if time.time()>deadline:raise TimeoutError('Prior queue/GPU still active after bounded six-hour wait')
            time.sleep(30)
        write(REC/(a.stage+'_frozen_inputs.json'),dict(files=frozen,commit=cfg['commit']))
        if a.stage=='preflight':
            invoke('gym_witness',[str(ROOT/'scripts/witness_pbhc_gym.py'),'--gpu','0','--output',str(REC/'gym_witness.json')])
            assert (REC/'gym_witness.json').exists()
            training('smoke128',128,2)
            provided=evaluate('provided',UP/cfg['provided_checkpoint'])
            training('capacity4096',4096,2)
            write(REC/'PREFLIGHT_COMPLETE.json',dict(provided_metrics=provided,full_training_started=False,
                next_gate='Fresh independent agent follows documented witness before --stage train'))
        else:
            checkpoint=training('official_horse_punch',4096,50000)
            provided=evaluate('provided_full',UP/cfg['provided_checkpoint'],1000)
            metrics=evaluate('retrained',checkpoint,1000)
            write(REC/'COMPLETE.json',dict(checkpoint=str(checkpoint),metrics=metrics,provided_metrics=provided,
                scope=cfg['scope'],environment_steps=cfg['total_environment_steps']))
        state.update(status='complete',finished=time.time());write(statepath,state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished=time.time());write(statepath,state);raise

if __name__=='__main__':main()
