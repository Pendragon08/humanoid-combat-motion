"""Freeze all matched training/evaluation inputs and launch a finite queue."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from combat.reference_bundle import build_bundle

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_matched_data_20260917'
OUT=ROOT/'runs/pbhc_matched_data_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','launch'],default='prepare');args=parser.parse_args()
    if args.stage=='launch':
        assert not (REC/'launch.json').exists(),'No duplicate launch'
        manifest=json.loads((REC/'manifest.json').read_text())
        for p,d in manifest['frozen_code_hashes'].items():assert sha(ROOT/p)==d,p
        assert sha(ROOT/'configs/pbhc_matched_data_pilot_v1.json')==manifest['protocol_sha256']
        approved=json.loads((ROOT/'records/pbhc_quality_gate_20260917/execution_manifest.json').read_text())
        for job in manifest['jobs']:
            prior=next(j for j in approved['jobs'] if j['name']==job['name'])
            for arm,key in [('raw','original_reference_sha256'),('quality_gated','reference_sha256')]:
                assert job['references'][arm]['sha256']==prior[key]
                assert sha(ROOT/job['references'][arm]['path'])==prior[key]
        states=[]
        for arm in ['raw','quality_gated']:
            folder=OUT/f'smoke_{arm}'
            receipt=json.loads((folder/'TRAINING_COMPLETE.json').read_text())
            assert receipt['environment_steps']==16*24*2
            states.append(json.loads((folder/'initialization.json').read_text()))
        for k in [k for k in states[0] if k.endswith('_sha256')]:assert states[0][k]==states[1][k]
        (REC/'prelaunch_validation.json').write_text(json.dumps(dict(smoke_training_steps_per_arm=768,
            matched_initial_digests=True,all_references_match_prior_execution=True,
            protocol_sha256=manifest['protocol_sha256'],launcher_sha256=sha(Path(__file__))),indent=2))
        with (REC/'scheduler.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'scripts/run_pbhc_matched_data.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (REC/'launch.json').write_text(json.dumps(dict(pid=child.pid,time=time.time()),indent=2))
        print('MATCHED_DATA_QUEUE_STARTED',child.pid,len(manifest['jobs']));return
    if REC.exists() or OUT.exists():raise FileExistsError('Preserve old queue, never duplicate')
    assert (ROOT/'records/pbhc_quality_gate_20260917/COMPLETE.json').exists()
    assert (ROOT/'records/pbhc_quality_gate_20260917/verification.json').exists()
    spec=json.loads((ROOT/'env-isaaclab.json').read_text())
    assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]=='072cb4cb'
    REC.mkdir(parents=True);OUT.mkdir(parents=True)
    cfg=json.loads((ROOT/'configs/pbhc_matched_data_pilot_v1.json').read_text())
    selection=json.loads((ROOT/'records/pbhc_quality_gate_20260917/selection.json').read_text())
    jobs=[]
    for gpu,name in enumerate(cfg['motions']):
        selected=next(r for r in selection['records'] if r['name']==name)
        assert selected['status']=='eligible_for_probe'
        refs={arm:ROOT/path for arm,path in dict(raw=f'runs/pbhc_beyondmimic_20260916/references_full/sample_{name}_actor0.npz',
                quality_gated=f'runs/pbhc_quality_gate_20260917/references/sample_{name}_actor0.npz').items()}
        assert sha(refs['raw'])==selected['source_sha256']
        for arm in cfg['arms']:
            bundle=OUT/'bundles'/f'{name}_{arm}.npz'
            build_bundle([dict(clip_id=name,source_id=selected['source_sha256'],path=str(refs[arm]),split='train',
                               evaluation_partition='development_matched_training_only',source_group=name)],bundle)
            jobs.append(dict(name=name,arm=arm,gpu=gpu,bundle=str(bundle.relative_to(ROOT)),bundle_sha256=sha(bundle),
                references={k:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for k,p in refs.items()},
                training_output=str((OUT/'training'/f'{name}_{arm}').relative_to(ROOT))))
    code=['scripts/train_pbhc_matched_data.py','scripts/run_pbhc_matched_data.py','scripts/summarize_pbhc_matched_data.py',
          'scripts/evaluate_pbhc_beyondmimic.py','scripts/evaluate_pbhc_followup.py','scripts/run_isaaclab.sh',
          'combat/reference_bundle.py','combat/multireference_tracking.py','combat/diagnostic_tracking.py']
    manifest=dict(config=cfg,jobs=jobs,environment_spec_hash='072cb4cb',
        frozen_code_hashes={p:sha(ROOT/p) for p in code},
        selection_sha256=sha(ROOT/'records/pbhc_quality_gate_20260917/selection.json'),
        protocol_sha256=sha(ROOT/'configs/pbhc_matched_data_pilot_v1.json'))
    (REC/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('MATCHED_DATA_PREPARED',len(jobs),'jobs; run matched two-update smoke before launch')

if __name__=='__main__':main()
