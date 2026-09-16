"""Read-only gates for the custom confirmation set, not official replication splits."""
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_manifest(root, manifest, annotations):
    root = Path(root)
    errors, records, labels = [], [], {}
    try:
        records = [json.loads(x) for x in Path(manifest).read_text(encoding='utf8').splitlines() if x.strip()]
    except (OSError, ValueError) as exc:
        errors.append('Cannot read manifest: ' + str(exc))
    try:
        with Path(annotations).open(encoding='utf8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for label in rows:
            key = label.get('annotation_id', '')
            if not key or key in labels:
                errors.append('Empty or duplicate annotation_id')
            labels[key] = label
    except OSError as exc:
        errors.append('Cannot read annotations: ' + str(exc))
    if not records:
        errors.append('Real motion manifest is empty; templates are not benchmark data')
    ids, groups, counts = set(), {}, {'train': 0, 'val': 0, 'test': 0}
    for r in records:
        sid = r.get('sample_id', '<missing>')
        prefix = str(sid) + ': '
        required = ('sample_id', 'dataset', 'split', 'source_ids', 'lineage_ids',
                    'source_sha256', 'motion_path', 'representation', 'fps', 'license_status')
        if any(k not in r for k in required):
            errors.append(prefix + 'missing provenance fields'); continue
        if sid in ids:
            errors.append(prefix + 'duplicate sample_id')
        ids.add(sid)
        split = r['split']
        if split not in counts:
            errors.append(prefix + 'invalid split'); continue
        counts[split] += 1
        if r['license_status'] != 'verified':
            errors.append(prefix + 'license not verified')
        if not isinstance(r['fps'], (int, float)) or not math.isfinite(r['fps']) or r['fps'] <= 0:
            errors.append(prefix + 'invalid fps')
        sources, lineage = r['source_ids'], r['lineage_ids']
        if not isinstance(sources, list) or not sources or not isinstance(lineage, list):
            errors.append(prefix + 'source/lineage must be lists; source nonempty'); continue
        for group in sources + lineage:
            if not isinstance(group, str) or not group:
                errors.append(prefix + 'invalid lineage group'); continue
            if group in groups and groups[group] != split:
                errors.append(prefix + 'cross-split source/descendant: ' + group)
            groups[group] = split
        if r.get('synthetic') and split == 'train' and r.get('conditioning_split') != 'train':
            errors.append(prefix + 'synthetic training condition not train')
        motion = root / r['motion_path']
        if not motion.is_file():
            errors.append(prefix + 'motion file absent')
        elif digest(motion) != r['source_sha256']:
            errors.append(prefix + 'motion checksum mismatch')
        if split != 'test':
            continue
        if r.get('test_use') != 'confirmation' or r.get('untouched') is not True:
            errors.append(prefix + 'not an untouched confirmation sample')
        if r.get('independent_annotation') is not True:
            errors.append(prefix + 'annotation is not independently verified')
        label = labels.get(r.get('annotation_id'))
        if not label or label.get('sample_id') != sid or label.get('split') != 'test':
            errors.append(prefix + 'matching test annotation absent'); continue
        if label.get('event_kind') not in ('Hit', 'Block', 'Avoid'):
            errors.append(prefix + 'unsupported event kind')
        if label.get('independent_of_generation_loss', '').lower() != 'true':
            errors.append(prefix + 'evaluation label duplicates generation loss')
        if not label.get('annotator_1') or not label.get('annotator_2') or label['annotator_1'] == label['annotator_2']:
            errors.append(prefix + 'two independent reviewers required')
        if label.get('agreement') not in ('agreed', 'adjudicated'):
            errors.append(prefix + 'annotation disagreement unresolved')
        if label.get('agreement') == 'adjudicated' and not label.get('adjudicator'):
            errors.append(prefix + 'adjudicator missing')
        if not label.get('evidence_path') or label.get('evidence_type') not in ('manual_review', 'synchronized_measurement'):
            errors.append(prefix + 'independent evidence absent')
        elif not (root / label['evidence_path']).exists():
            errors.append(prefix + 'independent evidence file absent')
        try:
            t = [float(label[k]) for k in ('start_s','window_start_s','window_end_s','end_s')]
            if not all(math.isfinite(v) for v in t) or not 0 <= t[0] <= t[1] < t[2] <= t[3]:
                raise ValueError()
        except (ValueError, KeyError):
            errors.append(prefix + 'invalid event time window')
    if not counts['test']:
        errors.append('No independent test samples')
    return {'schema_version': 1, 'confirmation_manifest_ready': not errors,
            'counts': counts, 'annotations': len(labels), 'errors': errors,
            'meaning': 'provenance consistency only; not annotation accuracy, statistical adequacy, or training authorization',
            'training_launched': False}
