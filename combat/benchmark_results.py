"""Parse completed upstream metric logs; uncertainty is over sampling repeats."""
import math
import re
import statistics


def parse_metric_log(text, arm, repeats):
    number = r'([-+0-9.eE]+)'
    prefix = r'---> \[' + re.escape(arm) + r'\] '
    values = {}
    for key in ('MM Distance', 'FID', 'Diversity', 'Multimodality'):
        values[key] = [float(v) for v in re.findall(prefix + re.escape(key) + ': ' + number, text)]
    for k in (1, 2, 3):
        lines = re.findall(prefix + r'R_precision: ([^\n]+)', text)
        values['R_precision_top'+str(k)] = [float(re.search(r'\(top '+str(k)+r'\): '+number, line).group(1)) for line in lines]
    for key, series in values.items():
        if len(series) != repeats or not all(math.isfinite(v) for v in series):
            raise ValueError('Incomplete or invalid metric: ' + key)
    return values


def summarize_series(values):
    n=len(values)
    return {'n':n,'mean':statistics.mean(values),
            'sample_std':statistics.stdev(values)if n>1 else None,
            'normal_95_halfwidth':1.96*statistics.stdev(values)/math.sqrt(n)if n>1 else None,
            'uncertainty_unit':'random generation/evaluation repetitions, not independent robot training seeds or source populations'}
