"""Map all four complete PBHC development references to named G1 29-DOF qpos.

No cropping, outcome-based selection, root correction or velocity scaling.
Run in the existing NumPy environment, then use IsaacLab FK conversion.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/pbhc_beyondmimic_20260916'


def main():
    dest = OUT / 'named_qpos'
    if dest.exists():
        raise FileExistsError(dest)
    inventory = json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())
    witness = json.loads((ROOT/'records/paired_v1/g1_independent_validation.json').read_text())
    names = witness['joint_names']
    assert len(names) == 29 and len(set(names)) == 29
    spec = json.loads((ROOT/'env-isaaclab.json').read_text())
    assert hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8] == '072cb4cb'
    dest.mkdir(parents=True)
    rows = []
    for case in inventory['motions']:
        if case['category'] != 'combat_candidate':
            continue
        path = Path(case['target_reference'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['target_sha256']
        with np.load(path, allow_pickle=False) as z:
            sn = list(z['joint_names']); pelvis = list(z['body_names']).index('pelvis')
            assert float(z['fps']) == 50 and len(sn) == 25
            q = np.zeros((len(z['joint_pos']), 36))
            q[:, :3] = z['body_pos_w'][:, pelvis]
            q[:, 3:7] = z['body_quat_w'][:, pelvis]
            added = []
            for j, name in enumerate(names):
                if name in sn:
                    q[:, 7+j] = z['joint_pos'][:, sn.index(name)]
                else:
                    assert 'wrist_pitch' in name or 'wrist_yaw' in name
                    added.append(name)
            assert len(added) == 4 and np.isfinite(q).all()
            assert np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-4)
        target = dest / ('sample_'+case['name']+'.npz')
        np.savez_compressed(target, qpos=q, fps=50., joint_names=np.array(names))
        rows.append(dict(name=case['name'], source=case, named_qpos=str(target),
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         frames=len(q), added_neutral_joints=added))
    assert len(rows) == 4
    (OUT/'source_manifest.json').write_text(json.dumps({
        'scope': 'All four full combat development examples; not independent testing',
        'mapping': '25 to 29 by name; four missing wrist joints zero; 23 original joints preserved',
        'reference_changes': 'No cropping, time scaling, root correction or joint alteration',
        'records': rows}, indent=2))
    print('PBHC_BEYONDMIMIC_NAMED_REFERENCES', len(rows))


if __name__ == '__main__':
    main()
