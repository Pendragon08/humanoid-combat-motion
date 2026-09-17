"""One bounded pre-launch API fix; retain failed smoke artifacts and old freeze."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_matched_data_20260917'
OUT=ROOT/'runs/pbhc_matched_data_20260917'

def main():
    assert not (REC/'launch.json').exists()
    old=REC/'manifest_before_api_fix.json'
    assert not old.exists(), 'No unbounded retries'
    shutil.copy2(REC/'manifest.json',old)
    for arm in ['raw','quality_gated']:
        folder=OUT/f'smoke_{arm}'
        assert folder.exists() and not (folder/'TRAINING_COMPLETE.json').exists()
        folder.rename(OUT/f'failed_api_smoke_{arm}')
        log=REC/f'smoke_{arm}.log'
        log.rename(REC/f'failed_api_smoke_{arm}.log')
    manifest=json.loads(old.read_text())
    trainer='scripts/train_pbhc_matched_data.py'
    manifest['frozen_code_hashes'][trainer]=hashlib.sha256((ROOT/trainer).read_bytes()).hexdigest()
    manifest['prelaunch_amendment']='RSL PPO uses policy, not actor_critic; include privileged normalizer. No successful training existed before fix; failed smoke artifacts retained.'
    (REC/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for arm in ['raw','quality_gated']:
        command=['bash','scripts/run_isaaclab.sh','scripts/train_pbhc_matched_data.py',
                 '--bundle',str(OUT/'bundles'/f'Horse-stance_punch_{arm}.npz'),
                 '--output',str(OUT/f'smoke_{arm}'),'--mode','upstream','--num_envs','16',
                 '--iterations','2','--seed','0','--headless','--device','cuda:0']
        if arm=='quality_gated':command+=['--match-initialization',str(OUT/'smoke_raw'/'initialization.json')]
        with (REC/f'smoke_{arm}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('MATCHED_SMOKE_REPAIR_COMPLETE')

if __name__=='__main__':main()
