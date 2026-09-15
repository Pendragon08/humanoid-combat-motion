"""Preserve small experiment records and log snapshots; exclude datasets/weights."""
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
record = root / 'records/tracking_diagnostics_20260915'
destination = record / 'snapshot'
allowed = {'config.json', 'TRAINING_COMPLETE.json', 'SMOKE_COMPLETE.json', 'metrics.json',
           'protocol.json', 'single.json', 'four.json', 'env.yaml', 'agent.yaml', 'provenance.json'}
entries = []
for directory in sorted((root / 'runs').glob('tracking_diagnostic*')):
    if not directory.is_dir():
        continue
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name not in allowed:
            continue
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError('Unexpectedly large record: ' + str(path))
        data = path.read_bytes()
        out = destination / directory.name / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        entries.append({'source': str(path.relative_to(root)), 'saved': str(out.relative_to(root)),
                        'sha256': hashlib.sha256(data).hexdigest()})
for path in sorted(record.glob('*.log')):
    data = path.read_bytes()
    out = destination / 'logs' / (path.name + '.gz')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(data, mtime=0))
    entries.append({'source': str(path.relative_to(root)), 'saved': str(out.relative_to(root)),
                    'uncompressed_sha256': hashlib.sha256(data).hexdigest(), 'snapshot_bytes': len(data)})
destination.mkdir(parents=True, exist_ok=True)
(destination / 'INDEX.json').write_text(json.dumps({'files': entries,
    'summary_sha256': hashlib.sha256((record / 'summary.json').read_bytes()).hexdigest(),
    'raw_motion_and_weights_included': False}, indent=2))
print('TRACKING_DIAGNOSTIC_RECORDS_SAVED', len(entries))
