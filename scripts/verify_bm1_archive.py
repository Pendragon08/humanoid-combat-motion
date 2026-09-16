"""Recompute BM1 metrics from archived logs without models, datasets, or GPUs."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from combat.benchmark_results import parse_metric_log, summarize_series


def same_summary(actual, expected):
    # Python versions can differ at the final floating-point rounding bit.
    return actual.keys() == expected.keys() and all(
        math.isclose(v, expected[k], rel_tol=1e-12, abs_tol=1e-12)
        if isinstance(v, float) else v == expected[k]
        for k, v in actual.items())


def verify(root, record):
    root, record = Path(root), Path(record)
    manifest = json.loads((root / 'configs/bm1_20260915_v1.json').read_text())
    result = json.loads((record / 'suite_result.json').read_text())
    index = json.loads((record / 'archive_index.json').read_text())
    for entry in index:
        payload = (record / entry['archive_path']).read_bytes()
        if entry['archive_path'].endswith('.gz'):
            payload = gzip.decompress(payload)
        if hashlib.sha256(payload).hexdigest() != entry['source_sha256']:
            raise ValueError('Archive checksum mismatch: ' + entry['archive_path'])
    for name, expected in manifest['files_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen execution code changed: ' + name)
    traces, inputs, metrics = {}, {}, {}
    for arm in manifest['jobs']:
        d = record / 'artifacts' / arm
        marker = json.loads((d / 'EVALUATION_COMPLETE.json').read_text())
        provenance = json.loads((d / 'provenance.json').read_text())
        log = (d / 'evaluation.log').read_bytes()
        trace = gzip.decompress((d / 'generation_inputs.jsonl.gz').read_bytes())
        frozen = (d / 'frozen_inputs.json').read_bytes()
        if not marker['complete'] or marker['repetitions'] != manifest['repetitions']:
            raise ValueError('Incomplete repetitions')
        if hashlib.sha256(log).hexdigest() != marker['log_sha256']:
            raise ValueError('Log differs from completion receipt')
        if hashlib.sha256(trace).hexdigest() != marker['trace_sha256']:
            raise ValueError('Trace differs from completion receipt')
        if hashlib.sha256(frozen).hexdigest() != provenance['frozen_inputs_sha256']:
            raise ValueError('Frozen input mismatch')
        rows = [json.loads(line) for line in trace.splitlines()]
        if len(rows) != marker['calls'] or any(r['call'] != i for i, r in enumerate(rows)):
            raise ValueError('Missing or reordered generation calls')
        if marker['calls'] != provenance['test_count'] * manifest['repetitions']:
            raise ValueError('Generation count mismatch')
        metrics[arm] = parse_metric_log(log.decode(), arm, manifest['repetitions'])
        if metrics[arm] != result['raw_metric_values'][arm]:
            raise ValueError('Reported raw metrics differ from logs')
        for name, values in metrics[arm].items():
            if not same_summary(summarize_series(values), result['arms'][arm][name]):
                raise ValueError('Summary does not reproduce: ' + name)
        traces[arm], inputs[arm] = trace, frozen
    if traces['baseline'] != traces['caption_adapter'] or inputs['baseline'] != inputs['caption_adapter']:
        raise ValueError('Paired conditions do not match')
    for name in metrics['baseline']:
        delta = [b-a for a,b in zip(metrics['baseline'][name], metrics['caption_adapter'][name])]
        if not same_summary(summarize_series(delta), result['paired_deltas'][name]):
            raise ValueError('Paired delta does not reproduce: ' + name)
    return {'verified': True, 'archived_files': len(index), 'repetitions_per_arm': manifest['repetitions'],
            'generation_calls_per_arm': len(rows), 'paired_inputs_and_rng': True,
            'metrics_recomputed_from_logs': True, 'frozen_execution_code_verified': True,
            'summary_float_tolerance': 1e-12,
            'scope': 'artifact and numerical verification, not independent contact or robot utility validation'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--record', default='records/bm1_20260915_v1')
    p.add_argument('--output')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = verify(root, root / args.record)
    text = json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        target = root / args.output
        with target.open('x', encoding='utf8') as stream:
            stream.write(text)
    print(text)
