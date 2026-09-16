"""Finite local watcher: archive only this run's receipts/logs and push authorized results.

No motion arrays, upstream code, weights, or unrelated staged changes are published.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--server',required=True)
    p.add_argument('--remote-root',required=True)
    a=p.parse_args();dest=a.repo/'records/remocap_20260916';dest.mkdir(parents=True,exist_ok=True)
    state=dest/'archive_watcher_status.json'
    ssh=['ssh','-i',str(a.identity),'-o','BatchMode=yes','-o','ConnectTimeout=15',a.server]
    remote=a.remote_root+'/records/remocap_20260916'
    # These paths are fixed application arguments, not text read from remote output.
    if not a.remote_root.startswith('/data/code/') or any(c in a.remote_root for c in "'\n\r; "):
        raise ValueError('Unexpected remote workspace')
    def write_status(status,**extra):
        state.write_text(json.dumps(dict(status=status,utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2))
    def run(cmd):
        return subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    start=time.monotonic();write_status('waiting_for_training_and_evaluation')
    completed=False;failure=None
    while time.monotonic()-start<6*3600:
        command=(f"cd {a.remote_root} && "
            "if test -f records/remocap_20260916/remos_EVALUATION_QUEUE_COMPLETE.json && "
            "test -f records/remocap_20260916/intergen_EVALUATION_QUEUE_COMPLETE.json; then echo COMPLETE; "
            "elif grep -q 'Traceback (most recent call last)' records/remocap_20260916/queue_*.log "
            "records/remocap_20260916/evaluation_queue_*.log; then echo FAILED; else echo RUNNING; fi")
        result=run(ssh+[command])
        if result.returncode==0 and 'COMPLETE' in result.stdout:
            completed=True;break
        if result.returncode==0 and 'FAILED' in result.stdout:
            failure='server_queue_failed';break
        time.sleep(45)
    if not completed and failure is None:failure='watcher_timeout'
    summary=run(ssh+[f'cd {a.remote_root} && bash scripts/run_timotion.sh -m scripts.summarize_remocap --records records/remocap_20260916'])
    if summary.returncode:write_status('summary_failed',error=summary.stderr[-2000:]);return
    names=['acquisition.json','data_audit.json','preprocess.log','model_smoke.log','protocol_tests.log',
           'regression_tests.log','regression_robot_env.log','suite_summary.json','suite_summary.md']
    for method in ['remos','intergen']:
        names += [f'pilot_{method}.log',f'queue_{method}.log',f'evaluation_queue_{method}.log',
                  f'{method}_TRAINING_QUEUE_COMPLETE.json',f'{method}_EVALUATION_QUEUE_COMPLETE.json']
        for seed in [0,1,2]:
            names += [f'{method}_seed{seed}.log',f'{method}_seed{seed}_training.json',
                      f'{method}_seed{seed}_evaluation.log',f'{method}_seed{seed}_evaluation.json']
    copied=[];missing=[]
    for name in names:
        result=run(['scp','-i',str(a.identity),'-o','BatchMode=yes',f'{a.server}:{remote}/{name}',str(dest/name)])
        if result.returncode==0:copied.append(name)
        else:missing.append(name)
    if completed and missing:
        write_status('archive_incomplete',missing=missing);return
    write_status('experiment_complete_archived' if completed else 'experiment_failed_archived',failure=failure,
                 missing=missing,raw_data_published=False,weights_published=False)
    (dest/'STATUS.md').write_text('# ReMoCap execution status\n\n'+
        ('Both 3-seed adapted baseline training/evaluation queues completed. See suite_summary.md.\n' if completed else
         'A queue failed or timed out. See the archived logs; do not treat partial output as completion.\n')+
        '\nThis does not establish independent contact labels, robot executability or downstream imitation benefit.\n')
    git=['git','-C',str(a.repo)]
    if run(git+['branch','--show-current']).stdout.strip()!='main':
        write_status('publish_blocked_branch_changed');return
    if run(git+['diff','--cached','--quiet']).returncode!=0:
        write_status('publish_blocked_unrelated_staged_changes');return
    paths=['records/remocap_20260916/'+n for n in copied+['archive_watcher_status.json','STATUS.md']]
    added=run(git+['add','--']+paths)
    if added.returncode:write_status('git_add_failed',error=added.stderr);return
    commit=run(git+['commit','-m','Archive ReMoCap adapted benchmark results and execution receipts'])
    if commit.returncode:write_status('git_commit_failed',error=commit.stderr+commit.stdout);return
    pushed=run(git+['push','origin','main'])
    # Keep publication status out of the committed watcher JSON to avoid a self-modifying commit loop.
    final={'published':pushed.returncode==0,'commit':run(git+['rev-parse','HEAD']).stdout.strip(),
           'stdout':pushed.stdout,'stderr':pushed.stderr}
    local=a.repo/'.aris/remocap_archive_publication.json';local.parent.mkdir(parents=True,exist_ok=True)
    local.write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)


if __name__=='__main__':main()
