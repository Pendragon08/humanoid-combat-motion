"""Summarize bounded native-runner comparisons without claiming data utility."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/engineai_official_v1'

def main():
    rows=[]
    for name in ['official_full_eval','raw_ik_prior_eval','raw_ik_eval','joint_constrained_prior_eval','joint_constrained_eval']:
        p=REC/(name+'_results.json')
        if not p.exists():continue
        result=json.loads(p.read_text())
        steps=result['first_failure_steps']
        rows.append({'condition':name,'environments':len(steps),'completed':round(result['completion_fraction']*len(steps)),'strict_pass':round(result['strict_tracking_fraction']*len(steps)),'reference_seconds':result['horizon']/50,'first_failure_or_completion_seconds_range':[min(steps)/50,max(steps)/50],**result['mean_alive_errors']})
    summary={'scope':'native official-framework reference-fit diagnostic; single training seed, one generated development motion','results':rows,'limitations':['32 environments are not 32 independent training seeds','errors are censored at first termination; smaller error alone is not better performance','official dance and generated combat differ in content and length','constraints are post-generation joint angle, speed and acceleration processing; not full dynamic constraints','no held-out motion test, no synthetic-data utility claim','no new standalone single-person generator was trained'],'storage':json.loads((REC/'storage_after.json').read_text())}
    (REC/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
