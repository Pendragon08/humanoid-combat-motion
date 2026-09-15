"""Audit actual seed files; structural integrity is not dynamics or data utility.

Uses the existing KungFuAthlete source manifest. Never changes files or splits.
No mocap, contact, cost or semantic ground truth is inferred from filenames.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_motion(path, expected_hash):
    result = {'file_sha256': sha(path), 'errors': [], 'warnings': []}
    result['hash_matches_manifest'] = result['file_sha256'] == expected_hash
    if not result['hash_matches_manifest']:
        result['errors'].append('source_hash_mismatch')
    with np.load(path, allow_pickle=False) as source:
        if 'qpos' not in source or 'fps' not in source:
            result['errors'].append('missing_qpos_or_fps')
            result['structural_valid'] = False
            return result
        q = np.asarray(source['qpos'], dtype=np.float64)
        fps_array = np.asarray(source['fps'])
    result['shape'] = list(q.shape)
    if fps_array.size != 1:
        result['errors'].append('nonscalar_fps')
        result['structural_valid'] = False
        return result
    fps = float(fps_array.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        result['errors'].append('invalid_fps')
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 3:
        result['errors'].append('expected_T_by_36_G1_free_root_plus_29_joints')
    if not np.isfinite(q).all():
        result['errors'].append('nonfinite_qpos')
    if result['errors']:
        result['structural_valid'] = False
        return result
    norm = np.linalg.norm(q[:, 3:7], axis=1)
    if (norm < 1e-8).any():
        result['errors'].append('zero_root_quaternion')
    if np.max(np.abs(norm-1)) > 1e-3:
        result['warnings'].append('requires_root_quaternion_normalization')
    velocity = np.diff(q[:, 7:], axis=0)*fps
    acceleration = np.diff(velocity, axis=0)*fps
    # Numeric equality diagnostic, not semantic near-duplicate detection.
    numeric_hash = hashlib.sha256(np.ascontiguousarray(q, dtype='<f8').tobytes())
    numeric_hash.update(np.asarray([fps], dtype='<f8').tobytes())
    result.update({
        'structural_valid': not result['errors'],
        'raw_reference_ready': not result['errors'] and not result['warnings'],
        'frames': len(q), 'fps': fps,
        'duration_s': (len(q)-1)/fps,
        'numeric_motion_sha256': numeric_hash.hexdigest(),
        'root_quaternion_norm_max_error': float(np.max(np.abs(norm-1))),
        'joint_speed_abs_max_rad_s': float(np.abs(velocity).max()),
        'joint_speed_abs_p99_rad_s': float(np.percentile(np.abs(velocity), 99)),
        'joint_acceleration_abs_max_rad_s2': float(np.abs(acceleration).max()),
        'joint_acceleration_abs_p99_rad_s2': float(np.percentile(np.abs(acceleration), 99)),
        'root_translation_speed_max_m_s': float(np.linalg.norm(np.diff(q[:, :3], axis=0)*fps, axis=1).max()),
        'root_height_min_m': float(q[:, 2].min()),
        'root_height_max_m': float(q[:, 2].max()),
        'joint_limit_checked': False, 'actuator_velocity_limit_checked': False,
        'dynamics_verified': False, 'paired_contact_verified': False,
        'scope': 'finite differences of saved references; no filtering by arbitrary speed thresholds'
    })
    return result


def duplicate_sets(rows):
    groups = defaultdict(list)
    for row in rows:
        key = row['quality'].get('numeric_motion_sha256')
        if key:
            groups[key].append(row)
    return [{'sample_ids': [r['sample_id'] for r in group],
             'splits': sorted({r['split'] for r in group}),
             'cross_split': len({r['split'] for r in group}) > 1}
            for group in groups.values() if len(group) > 1]


def audit(root, output):
    start = time.perf_counter()
    root = root.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = root/'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows, ids = [], set()
    source_splits = defaultdict(set)
    for source in manifest['motions']:
        sample_id = 'kungfu_'+str(source['id'])
        if sample_id in ids:
            raise ValueError('Duplicate source ID: '+sample_id)
        ids.add(sample_id)
        if source['kind'] != 'video_reconstructed_and_GMR_retargeted':
            raise ValueError('Unexpected acquisition type; do not silently relabel: '+sample_id)
        split = 'val' if source['split'] == 'validation' else source['split']
        if split not in ('train', 'val', 'test') or not source.get('source_group'):
            raise ValueError('Missing/invalid source split: '+sample_id)
        source_splits[source['source_group']].add(split)
        path = (root/source['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source path outside project')
        quality = inspect_motion(path, source['sha256']) if path.is_file() else {
            'structural_valid': False, 'errors': ['missing_file'], 'warnings': []}
        rows.append({
            'sample_id': sample_id, 'source_type': 'V', 'representation': 'R_G1_29dof',
            'split': split, 'original_split_name': source['split'], 'source_group': source['source_group'],
            'original_source_id': str(source['id']), 'title': source['title'],
            'source_path': source['path'], 'source_url': source['url'],
            'hf_revision': manifest['hf_revision'], 'parent_sha256': source['sha256'],
            'license_statement': manifest['dataset_license'],
            'license_evidence': 'data/DATASET_CARD.md at locally recorded revision',
            'upstream_video_rights_verified': False, 'paired': source['paired'],
            'performer_identity': None, 'combat_semantics_verified': False,
            'role_label': None, 'contact_label': None, 'contact_confidence': None,
            'acquisition_cost': None, 'cleaning_minutes': None, 'annotation_minutes': None,
            'generation_gpu_hours': None, 'cost_missing_reason': 'not recorded by source manifest',
            'file_bytes': path.stat().st_size if path.is_file() else None,
            'quality': quality})
    overlap = {g: sorted(s) for g,s in source_splits.items() if len(s)>1}
    duplicates = duplicate_sets(rows)
    groups = {}
    for split in ('train','val','test'):
        chosen = [r for r in rows if r['split']==split]
        groups[split] = {
            'clips': len(chosen), 'source_groups': len({r['source_group'] for r in chosen}),
            'structural_valid_clips': sum(r['quality']['structural_valid'] for r in chosen),
            'clip_duration_sum_s': sum(r['quality'].get('duration_s',0) for r in chosen),
            'groups': sorted({r['source_group'] for r in chosen})}
    valid = [r for r in rows if r['quality']['structural_valid']]
    review = sorted(rows, key=lambda r: (not r['quality']['structural_valid'],
                     r['quality'].get('joint_speed_abs_max_rad_s',0)), reverse=True)
    summary = {
        'experiment': 'E1 seed source and kinematic integrity audit',
        'status': 'completed_partial_E1_not_full_quality_or_utility_validation',
        'execution_host': platform.node(), 'execution_platform': platform.platform(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'manifest_sha256': sha(manifest_path), 'dataset_card_sha256': sha(root/'data/DATASET_CARD.md'),
        'script_sha256': sha(Path(__file__)), 'input_files': len(rows),
        'structural_valid_files': len(valid), 'split_summary': groups,
        'raw_reference_ready_files': sum(r['quality'].get('raw_reference_ready',False) for r in rows),
        'quaternion_normalization_required_files': sum('requires_root_quaternion_normalization' in r['quality']['warnings'] for r in rows),
        'normalization_caveat': 'existing combat.robot.Robot.remap already normalizes exported references; this raw audit does not establish a prior training bug',
        'source_group_overlap': overlap, 'numeric_duplicate_groups': duplicates,
        'numeric_cross_split_duplicates': sum(x['cross_split'] for x in duplicates),
        'errors': dict(Counter(e for r in rows for e in r['quality']['errors'])),
        'total_bytes': sum(r['file_bytes'] or 0 for r in rows),
        'clip_duration_sum_s': sum(r['quality'].get('duration_s',0) for r in rows),
        'duration_caveat': 'sum of subclip spans; not guaranteed unique original video time',
        'sensor_mocap_files_verified': 0, 'paired_files': sum(r['paired'] for r in rows),
        'manual_combat_or_contact_verified_files': 0, 'cost_complete_files': 0,
        'median_clip_peak_joint_speed_rad_s': float(np.median([r['quality']['joint_speed_abs_max_rad_s'] for r in valid])) if valid else None,
        'max_clip_peak_joint_speed_rad_s': max((r['quality']['joint_speed_abs_max_rad_s'] for r in valid),default=None),
        'runtime_s': time.perf_counter()-start,
        'next_required': ['independent semantic/contact review', 'source cost evidence',
                          'paired human data audit', 'new untouched source groups',
                          'robot-model-specific limits and dynamics checks'],
        'claims_not_supported': ['sensor MoCap replacement', 'paired combat dataset',
                                 'actuator feasibility', 'imitation learning improvement'],
        'source_modified': False, 'new_training_jobs': 0}
    output.mkdir(parents=True)
    def save(name, data):
        (output/name).write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    save('summary.json', summary)
    (output/'manifest.jsonl').write_bytes((''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)).encode('utf-8'))
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer, fieldnames=['sample_id','split','source_group','structural_valid','peak_joint_speed_rad_s','semantic_label','contact_label','review_status'])
    writer.writeheader()
    for row in review:
        writer.writerow({'sample_id':row['sample_id'],'split':row['split'],'source_group':row['source_group'],
                         'structural_valid':row['quality']['structural_valid'],
                         'peak_joint_speed_rad_s':row['quality'].get('joint_speed_abs_max_rad_s'),
                         'semantic_label':'','contact_label':'','review_status':'not_reviewed'})
    (output/'review_queue.csv').write_bytes(buffer.getvalue().replace('\r\n','\n').encode('utf-8'))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    audit(a.root,a.output)
