"""Freeze training-only references for interface/sampling diagnostics, not semantic combat validation."""
from pathlib import Path
import json
import hashlib
import numpy as np
from combat.reference_bundle import build_bundle

root = Path(__file__).resolve().parents[1]
out = root/'runs/tracking_diagnostics_20260915_refs'
if out.exists():
    raise FileExistsError(out)
entries = json.loads((root/'data/tracking_seed/kungfu_train20_bundle.json').read_text())['clips']
# First four distinct source-title groups, first existing training clip per group.
# No selection using prior tracking scores or test outcomes.
selected = []
groups = set()
for e in entries:
    if e['source_group'] not in groups:
        selected.append(e); groups.add(e['source_group'])
    if len(selected) == 4:
        break
assert len(selected) == 4
audits = []
for e in selected:
    path=Path(e['path'])
    assert e['split']=='train' and hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
    with np.load(path,allow_pickle=False) as z:
        assert float(z['fps'])==50 and len(z['joint_names'])==29
        for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'):
            assert np.isfinite(z[key]).all()
        assert np.max(np.abs(np.linalg.norm(z['body_quat_w'],axis=-1)-1))<1e-3
        audits.append({'clip_id':e['clip_id'],'frames':len(z['joint_pos']),
                       'duration_s':len(z['joint_pos'])/50,'peak_joint_speed_rad_s':float(np.abs(z['joint_vel']).max()),
                       'source_group':e['source_group'],'semantics_manually_verified':False})
out.mkdir(parents=True)
build_bundle(selected[:1],out/'single.npz')
build_bundle(selected,out/'four.npz')
(out/'evaluate_training_paths.txt').write_text('\n'.join(e['path'] for e in selected)+'\n')
record={'selection':'first training clip of first four distinct title groups, original manifest order',
        'scope':'interface and clip-sampling diagnosis; not manually verified punch/block classes or a low-dynamic benchmark',
        'references':selected,'audits':audits,'seed':0,'updates':1000,'envs':512,'steps_per_env':24,
        'jobs':['single_upstream','single_uniform','four_uniform','four_adaptive'],
        'evaluation':'32 environments per reference, reset seed 123; all four selected training references and four existing development holdouts',
        'changes':'upstream-vs-single_uniform includes reset/termination/alignment differences, not isolated sampling effect',
        'environment_spec_sha256_8':hashlib.sha256(json.dumps(json.loads((root/'env-isaaclab.json').read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]}
assert record['environment_spec_sha256_8']=='072cb4cb'
(out/'protocol.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print('DIAGNOSTIC_REFERENCES_READY',json.dumps(audits,ensure_ascii=False),flush=True)
