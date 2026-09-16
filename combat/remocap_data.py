"""Recording-preserving body data conversion; no imported pickle execution."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

JOINTS = ['Hips','LeftUpLeg','RightUpLeg','Spine','LeftLeg','RightLeg','Spine1',
          'LeftFoot','RightFoot','Spine3','LeftToeBase','RightToeBase','Neck',
          'LeftShoulder','RightShoulder','Head','LeftArm','RightArm',
          'LeftForeArm','RightForeArm','LeftHand','RightHand']
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_actor(path):
    with path.open() as f:
        header = next(csv.reader(f))
    columns = [header.index(j + '.' + axis) for j in JOINTS for axis in 'XYZ']
    data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[0] + columns)
    if not np.isfinite(data).all() or len(data) < 3:
        raise ValueError('Invalid motion: ' + str(path))
    times = data[:, 0]
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, np.median(dt), atol=1e-5):
        raise ValueError('Irregular timestamps require explicit repair')
    bvh = path.with_name(path.name.split('_')[0] + '.bvh')
    with bvh.open() as f:
        for line in f:
            if 'Frame Time:' in line:
                bvh_dt = float(line.split(':')[1])
                break
        else:
            raise ValueError('Missing BVH frame time')
    if not np.isclose(bvh_dt, np.median(dt), atol=1e-5):
        raise ValueError('CSV/BVH time mismatch')
    positions = data[:, 1:].reshape(-1, 22, 3).astype(np.float32) / 1000.
    shank = np.linalg.norm(positions[:, 4] - positions[:, 7], axis=-1)
    if not .15 < np.median(shank) < .8:
        raise ValueError('Unexpected units/anatomy: ' + str(path))
    return times, positions, 1 / np.median(dt)


def common_resample(t0, p0, t1, p1, fps):
    start, end = max(t0[0], t1[0]), min(t0[-1], t1[-1])
    t = start + np.arange(int(np.floor((end-start)*fps + 1e-6))+1) / fps
    actors = []
    for original_t, pos in [(t0, p0), (t1, p1)]:
        flat = pos.reshape(len(pos), -1)
        actors.append(np.stack([np.interp(t, original_t, flat[:, j]) for j in range(66)], axis=-1).reshape(-1,22,3))
    return np.stack(actors, axis=1).astype(np.float32)


def canonicalize(pair):
    """Only supplied actor's first frame determines translation and yaw."""
    out = pair.copy()
    origin = pair[0,0,0].copy()
    origin[1] = 0  # Preserve floor-relative height.
    across = (pair[0,0,2]-pair[0,0,1]) + (pair[0,0,14]-pair[0,0,13])
    across[1] = 0
    if np.linalg.norm(across) < 1e-6:
        raise ValueError('Degenerate actor orientation')
    right = across/np.linalg.norm(across)
    up = np.array([0.,1.,0.])
    forward = np.cross(right, up)
    rotation = np.stack([right, up, forward])
    out = np.einsum('ij,takj->taki', rotation, out-origin).astype(np.float32)
    return out


def prepare(root, output, receipt_path, frames=40, fps=20):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for split in ['train', 'test']:
        for folder in sorted((root/split).iterdir()):
            if not folder.is_dir():
                continue
            files = [folder/f'{i}_worldpos.csv' for i in range(2)]
            if not all(p.exists() for p in files):
                raise ValueError('Missing paired CSV: '+str(folder))
            rows.append(dict(official_split=split, source=folder.name,
                             paths=files, hashes=[digest(p) for p in files]))
    train_names = {r['source'] for r in rows if r['official_split']=='train'}
    test_names = {r['source'] for r in rows if r['official_split']=='test'}
    if train_names & test_names:
        raise ValueError('Source identity overlaps official train/test')
    ranked = sorted(train_names, key=lambda s: hashlib.sha256(('val-v2:'+s).encode()).hexdigest())
    val_names = set(ranked[:max(1, round(.2*len(ranked)))])
    seen_files, seen_windows = {}, {}
    windows = {s: [] for s in ['train','val','test']}
    window_metadata = {s: [] for s in windows}
    audits = []
    for row in rows:
        split = 'val' if row['source'] in val_names else row['official_split']
        for h in row['hashes']:
            if h in seen_files and seen_files[h] != split:
                raise ValueError('Cross-split duplicate motion')
            seen_files[h] = split
        t0,p0,f0 = read_actor(row['paths'][0])
        t1,p1,f1 = read_actor(row['paths'][1])
        pair = common_resample(t0,p0,t1,p1,fps)
        count = 0
        # Non-overlapping windows. Never cross recording boundaries.
        for start in range(0, len(pair)-frames+1, frames):
            x = canonicalize(pair[start:start+frames])
            h = hashlib.sha256(x.tobytes()).hexdigest()
            if h in seen_windows:
                if seen_windows[h] != split:
                    raise ValueError('Cross-split duplicate window')
                continue
            seen_windows[h] = split
            windows[split].append(x)
            window_metadata[split].append(dict(source=row['source'], start_frame=start, sha256=h))
            count += 1
        audits.append(dict(source=row['source'], split=split, official_split=row['official_split'],
            original_fps=[f0,f1], original_frames=[len(t0),len(t1)], resampled_frames=len(pair),
            windows=count, source_sha256=row['hashes']))
    output.mkdir(parents=True)
    train = np.stack(windows['train'])
    mean = train.mean(axis=(0,1,2), keepdims=False)
    std = train.std(axis=(0,1,2), keepdims=False).clip(.1)
    np.savez(output/'normalization.npz', mean=mean, std=std)
    for split in windows:
        array = np.stack(windows[split])
        np.save(output/f'{split}.npy', array)
        (output/f'{split}_windows.json').write_text(json.dumps(window_metadata[split],indent=2))
    receipt = dict(protocol='remocap_body_reaction_v2_20260916', fps=fps, frames=frames,
        joints=JOINTS, parents=PARENTS, units='m', up_axis='Y',
        window_counts={k:len(v) for k,v in windows.items()},
        source_counts={s:sum(r['split']==s for r in audits) for s in windows},
        source_audit=audits, cross_split_exact_duplicates=False,
        subject_disjoint_verified=False, source_disjoint=True,
        normalization_fit='train only', transformation='actor first frame only',
        attack_defense_labels_available=False, measured_contact_labels_available=False,
        upstream_preprocessor_deviation='Read actual CSV/BVH timestamps rather than hardcoded 25 fps; no contact-based frame deletion or sequence concatenation',
        output_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:receipt[k] for k in ['source_counts','window_counts','fps','frames']}),flush=True)
    return receipt
