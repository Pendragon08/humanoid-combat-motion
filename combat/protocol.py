"""Fail-closed experiment provenance and split checks."""
import hashlib
import json
import math
from pathlib import Path


def source_split_check(records):
    seen, ids = {}, set()
    for r in records:
        if r['sample_id'] in ids:
            raise ValueError('Duplicate sample_id: '+r['sample_id'])
        ids.add(r['sample_id'])
        split = r['split']
        if split not in ('train','val','test'):
            raise ValueError('Unknown split')
        if not r.get('source_ids'):
            raise ValueError('Every sample needs original source_ids')
        for source in r['source_ids']:
            if source in seen and seen[source] != split:
                raise ValueError('Source leakage: '+source)
            seen[source] = split
        if r.get('synthetic') and split == 'train' and r.get('conditioning_split') != 'train':
            raise ValueError('Synthetic training motion must originate from train conditions')
    return {'samples':len(records),'sources':len(seen),'source_disjoint':True}


def validate_physics_label(record):
    if record.get('critic_training_labels_approved') is not True:
        raise ValueError('Unapproved diagnostic records cannot train the full physics critic')
    if record.get('contact_observation_complete') is not True or record.get('full_collision_audit') is not True:
        raise ValueError('Contact censoring/full collision coverage must be audited')
    required = ('sample_id','source_ids','split','simulator','simulator_version','robot_model_sha256',
                'controller_kind','controller_checkpoint_sha256','seed','tracking_error_m','fall',
                'contact_success','invalid_collision','rollout_path','rollout_sha256')
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError('Missing physics provenance: '+','.join(missing))
    if record['controller_kind'] != 'learned_tracking':
        raise ValueError('PD/open-loop diagnostics are not learned-tracker critic labels')
    if record.get('robot_actor_count') != 2:
        raise ValueError('Dual-robot critic requires two-robot rollout labels')
    if not math.isfinite(float(record['tracking_error_m'])) or record['tracking_error_m']<0:
        raise ValueError('Invalid tracking error')
    for name in ('fall','contact_success','invalid_collision'):
        if record[name] not in (0,1,False,True):
            raise ValueError('Invalid binary target: '+name)
    if record['simulator'] not in ('IsaacLab','MuJoCo'):
        raise ValueError('Unrecognized simulator')
    if len(record['source_ids']) == 0:
        raise ValueError('Missing source lineage')
    for key in ('robot_model_sha256','controller_checkpoint_sha256','rollout_sha256'):
        if len(record[key]) != 64 or any(c not in '0123456789abcdef' for c in record[key]):
            raise ValueError('Invalid SHA256: '+key)
    return True


def check_artifact(path, digest):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2), b''):
            h.update(block)
    if h.hexdigest() != digest:
        raise ValueError('Artifact hash mismatch: '+str(path))
    return True


def load_manifest(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    source_split_check(records)
    return records
