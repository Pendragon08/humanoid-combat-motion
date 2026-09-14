"""Refresh current status without rewriting historical result records."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
summary=json.loads((root/'records/step7/summary.json').read_text())
path=root/'records/paired_v1/execution_status_20260914.json'
status=json.loads(path.read_text())
running=[]
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():continue
    try:
        args=[x.decode() for x in (directory/'cmdline').read_bytes().split(b'\0') if x]
        if not (directory/'exe').resolve(strict=True).name.startswith('python'):continue
    except (OSError,UnicodeDecodeError):continue
    names={Path(arg).name for arg in args}
    if not names & {'train_multireference_tracking.py','evaluate_g1_tracking.py'} or '--output' not in args:continue
    output=args[args.index('--output')+1]
    if not output.startswith('runs/step7_'):continue
    running.append({'pid':int(directory.name),'output':output,'kind':'training' if 'train_multireference_tracking.py' in names else 'evaluation'})
status['running']=running
status['completed']['unit_tests']='38 passed, including coverage feature height/sign integrity and physics-label refusal'
status['completed']['step7_coverage_diagnostic']='Three predictor seeds and K=6 retrospective ranking completed; not approved as a physics feasibility filter'
status['step7']={'completed_policy_runs':summary['completed_runs'],'expected_policy_runs':9,
    'component_utility_pilot_complete':summary['pilot_runs_complete'],'full_physics_pipeline_validated':False,
    'report':'records/step7/summary.json','contact_retargeting':'20 train cases; geometric contact 19/20 in humans and 4/20 with toe helper robot proxy; not force certification'}
if summary['pilot_runs_complete']:
    status['not_completed']=[x for x in status['not_completed'] if x!='equal-budget >=3-seed downstream utility and generalization comparisons']
    item='confirmatory utility of physics-critic-filtered paired combat data on independent source groups; current 3x3 pilot is contact-only, single-robot, one development video group'
    if item not in status['not_completed']:status['not_completed'].append(item)
    status['scientific_conclusion']='Nine equal-budget component utility runs completed; see step7 summary for measured effects. Full physics filtering and paired combat executability remain unvalidated; do not claim full plan success.'
path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print('STEP7_PROJECT_STATUS',summary['completed_runs'],'of9;',len(running),'active training/evaluation processes',flush=True)
