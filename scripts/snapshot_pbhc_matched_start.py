"""Archive startup evidence without modifying the running queue or its freeze."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_matched_data_20260917'
OUT=ROOT/'runs/pbhc_matched_data_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=json.loads((REC/'manifest.json').read_text());cfg=manifest['config']
    assert not (REC/'startup_evidence.json').exists()
    for rel,digest in manifest['frozen_code_hashes'].items():assert sha(ROOT/rel)==digest
    tests=[]
    for arm in cfg['arms']:
        path=OUT/f'smoke_{arm}';config=json.loads((path/'config.json').read_text())
        job=next(j for j in manifest['jobs'] if j['name']=='Horse-stance_punch' and j['arm']==arm)
        assert config['bundle_sha256']==job['bundle_sha256'] and config['seed']==0
        assert config['iterations']==2 and config['environments']==16 and config['steps_per_env']==24
        assert json.loads((path/'TRAINING_COMPLETE.json').read_text())['environment_steps']==768
        tests.append(dict(arm=arm,config=config,checkpoint_receipt=json.loads((path/'model_1.pt.receipt.json').read_text())))
        dest=REC/'smoke'/arm;dest.mkdir(parents=True)
        for file in ['config.json','initialization.json','TRAINING_COMPLETE.json','model_1.pt.receipt.json']:
            shutil.copy2(path/file,dest/file)
    for job in [j for j in manifest['jobs'] if j['arm']=='raw']:
        path=ROOT/job['training_output'];actual=json.loads((path/'config.json').read_text())
        assert actual['steps_per_env']==24 and actual['environments']==512 and actual['iterations']==1000
        assert actual['bundle_sha256']==job['bundle_sha256']
        dest=REC/'startup'/job['name'];dest.mkdir(parents=True)
        for file in ['config.json','initialization.json','progress.json']:shutil.copy2(path/file,dest/file)
    lib=ROOT/'third_party/isaacsim-4.5.0/kit/python/lib/python3.10/site-packages'
    paths=[lib/'rsl_rl/algorithms/ppo.py',lib/'rsl_rl/runners/on_policy_runner.py']
    paths+=list((ROOT/'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking').rglob('*.py'))
    result=dict(time=time.time(),scope='Post-launch startup witness; does not retroactively pre-freeze runtime dependencies',
        smoke=tests,frozen_code_still_matches=True,actual_raw_training_budget_matches=True,
        current_runtime_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        limitation='Startup smoke logs and current frozen-script hash retained; trainer digest was not embedded directly in original smoke receipt')
    (REC/'startup_evidence.json').write_text(json.dumps(result,indent=2))
    print('MATCHED_STARTUP_VERIFIED')

if __name__=='__main__':main()
