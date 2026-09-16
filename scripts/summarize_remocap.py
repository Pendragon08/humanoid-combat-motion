"""Aggregate seeds and recording-level uncertainty; never use frames as replicates."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True)
    a=p.parse_args();results={}
    for method in ['remos','intergen']:
        files=[a.records/f'{method}_seed{s}_evaluation.json' for s in [0,1,2]]
        if not all(f.exists() for f in files):
            results[method]={'status':'evaluation_incomplete'};continue
        records=[json.loads(f.read_text()) for f in files]
        out={}
        for track in ['full_test_raw','source_balanced_ablation']:
            out[track]={}
            for arm in records[0]['aggregate'][track]:
                sources=sorted(records[0]['aggregate'][track][arm]['source_means'])
                keys=records[0]['aggregate'][track][arm]['source_macro_mean']
                arm_result={}
                for metric in keys:
                    values=np.array([[r['aggregate'][track][arm]['source_means'][s][metric] for s in sources] for r in records])
                    baseline=np.array([[r['aggregate'][track]['raw']['source_means'][s][metric] for s in sources] for r in records])
                    rng=np.random.default_rng(20260916)
                    estimates=[];deltas=[]
                    for _ in range(2000):
                        seeds=rng.integers(0,3,3);ids=rng.integers(0,len(sources),len(sources))
                        estimates.append(values[seeds][:,ids].mean())
                        deltas.append((values-baseline)[seeds][:,ids].mean())
                    arm_result[metric]=dict(mean=float(values.mean()),seed_means=values.mean(1).tolist(),
                        crossed_seed_source_bootstrap_95=np.percentile(estimates,[2.5,97.5]).tolist(),
                        delta_vs_raw=float((values-baseline).mean()),paired_delta_95=np.percentile(deltas,[2.5,97.5]).tolist())
                out[track][arm]=arm_result
        results[method]=dict(status='complete',seeds=[0,1,2],metrics=out)
    report=dict(results=results,official_paper_reproduction=False,
        independent_contact_verified=False,robot_utility_verified=False,
        uncertainty='Exploratory crossed bootstrap of 3 training seeds and held-out recordings; no frame pseudoreplication',
        scope='Body-only adapted baselines and retrieval-conditioned refinement components; not full proposed method')
    (a.records/'suite_summary.json').write_text(json.dumps(report,indent=2))
    lines=['# ReMoCap / Ninjutsu adapted body benchmark','',report['scope'],'',
        '| Method | Arm | MPJPE (mm) | Velocity error (mm/s) | Acceleration RMS (m/s²) |',
        '|---|---|---:|---:|---:|']
    for method,r in results.items():
        if r['status']!='complete':lines.append(f'| {method} | pending | — | — | — |');continue
        for arm,metrics in r['metrics']['source_balanced_ablation'].items():
            lines.append(f"| {method} | {arm} | {metrics['mpjpe_mm']['mean']:.2f} | {metrics['velocity_error_mm_s']['mean']:.2f} | {metrics['acceleration_rms_m_s2']['mean']:.2f} |")
    lines+=['','Table uses the fixed source-balanced component subset, not all windows. Full raw-test results and uncertainty are in suite_summary.json.','',
        'No official FID, verified hit/block/contact force, or robot imitation benefit is established.']
    (a.records/'suite_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['status'] for k,v in results.items()}))


if __name__=='__main__':main()
