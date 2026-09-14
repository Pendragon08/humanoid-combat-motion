"""Read-only numerical audit for the revised plan; no causal inference or training."""
from pathlib import Path
import hashlib
import json
import statistics

root = Path(__file__).resolve().parents[1]
inputs = {}
def read(relative):
    data = (root / relative).read_bytes()
    inputs[relative] = hashlib.sha256(data).hexdigest()
    return json.loads(data)

summary = read('records/step7/summary.json')
assert summary['pilot_runs_complete'] and summary['completed_runs'] == 9
groups = {}
for arm, item in summary['groups'].items():
    values = [r['macro']['strict'] for r in summary['rows'] if r['arm'] == arm]
    assert len(values) == 3
    groups[arm] = {'strict_per_seed': values,
                   'strict_mean': statistics.mean(values),
                   'strict_sample_std': statistics.stdev(values),
                   'completion_mean': item['metrics']['completion']['mean'],
                   'time_to_failure_or_completion_s': item['metrics']['failure_time_s']['mean']}
base = groups['real']['strict_mean']
for arm in ('generic', 'contact'):
    groups[arm]['relative_strict_change_vs_real'] = groups[arm]['strict_mean'] / base - 1

retargeting = {}
for arm in ('generic', 'contact'):
    m = read(f'records/step7/snapshot/step7_{arm}_g1ik/manifest.json')
    speeds = [r['max_joint_speed_rad_s'] for r in m['samples']]
    retargeting[arm] = {'cases': len(speeds), 'per_case_peak_speed_rad_s': speeds,
                        'peak_speed_min_median_max_rad_s': [min(speeds), statistics.median(speeds), max(speeds)],
                        'hardware_velocity_violation_fraction': None,
                        'reason': 'No per-joint actuator velocity certification in this audit; peak speed is not a violation label.',
                        'method': m['method']}

report = {'status': 'observational_audit_not_new_experiment', 'groups': groups,
          'retargeting': retargeting, 'inputs_sha256': inputs,
          'causal_attribution_established': False,
          'caveats': ['One development video group; three controller seeds are not three data sources.',
                      'Identical total environment steps do not match real-source exposure.',
                      'C changes adapter and contact optimization together.',
                      'A/B/C downstream tracking is single-robot, not paired interaction.']}
out = root / 'records/step7/bottleneck_audit.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
