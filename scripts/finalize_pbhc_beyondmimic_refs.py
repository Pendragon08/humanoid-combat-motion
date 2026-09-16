"""Audit IsaacLab FK conversion and make one-clip development training bundles."""
import hashlib
import json
from pathlib import Path
import numpy as np
from combat.reference_bundle import build_bundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/pbhc_beyondmimic_20260916'
source = json.loads((OUT/'source_manifest.json').read_text())
audits = []
for row in source['records']:
    name = row['name']; ref = OUT/'references_full'/('sample_'+name+'_actor0.npz')
    with np.load(ref, allow_pickle=False) as z, np.load(row['named_qpos'], allow_pickle=False) as original:
        assert len(z['joint_pos']) == row['frames'], (name, len(z['joint_pos']), row['frames'])
        assert float(np.asarray(z['fps']).item()) == 50
        order = [list(original['joint_names']).index(n) for n in z['joint_names']]
        joint_error = float(abs(z['joint_pos']-original['qpos'][:, 7:][:, order]).max())
        pelvis = list(z['body_names']).index('pelvis')
        root_error = float(abs(z['body_pos_w'][:, pelvis]-original['qpos'][:, :3]).max())
        assert pelvis == 0 and joint_error < 1e-4 and root_error < 1e-4
        audits.append(dict(name=name, frames=len(z['joint_pos']), duration_s=(len(z['joint_pos'])-1)/50,
                           joint_conversion_max_error_rad=joint_error, root_conversion_max_error_m=root_error,
                           peak_joint_speed_rad_s=float(abs(z['joint_vel']).max()),
                           reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()))
    # "train" is the bundle loader's technical label. All sources remain development-only.
    entry=dict(clip_id=name, source_id=row['source']['source_sha256'], path=str(ref), split='train',
               evaluation_partition='development_calibration_only', source_group=name)
    build_bundle([entry], OUT/'bundles'/(name+'.npz'))
(OUT/'conversion_audit.json').write_text(json.dumps({'records':audits, 'physics_validated':False}, indent=2))
print('PBHC_BEYONDMIMIC_FK_AUDIT_PASS', json.dumps(audits))
