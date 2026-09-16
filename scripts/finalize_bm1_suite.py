"""Wait for both authorized jobs, verify paired traces, save comparison; no reruns."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
from combat.benchmark_results import parse_metric_log, summarize_series


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];m=json.loads(a.manifest.read_text())
    out=root/'records'/m['run_id'];target=out/'suite_result.json'
    if target.exists():raise FileExistsError(target)
    deadline=time.monotonic()+m['timeout_seconds']+1800
    while True:
        states={}
        for arm,job in m['jobs'].items():
            q=root/job['directory']/'status.json'
            states[arm]=json.loads(q.read_text()) if q.exists()else {'status':'not_started'}
        if all(s['status']=='complete'for s in states.values()):break
        if any(s['status'].startswith(('failed','blocked','timed_out'))for s in states.values())or time.monotonic()>deadline:
            target.write_text(json.dumps({'status':'incomplete','states':states,'comparison_claim':False},indent=2));return
        time.sleep(30)
    metrics={};traces={};provenance={};sources={}
    for arm,job in m['jobs'].items():
        d=root/job['directory']/'evaluation'
        marker=json.loads((d/'EVALUATION_COMPLETE.json').read_text())
        for name,key in [('evaluation.log','log_sha256'),('generation_inputs.jsonl','trace_sha256')]:
            if hashlib.sha256((d/name).read_bytes()).hexdigest()!=marker[key]:raise ValueError('Output checksum mismatch')
        provenance[arm]=json.loads((d/'provenance.json').read_text())
        if hashlib.sha256((d/'frozen_inputs.json').read_bytes()).hexdigest()!=provenance[arm]['frozen_inputs_sha256']:raise ValueError('Input receipt mismatch')
        sources[arm]=provenance[arm]['frozen_inputs_sha256']
        metrics[arm]=parse_metric_log((d/'evaluation.log').read_text(),arm,m['repetitions'])
        traces[arm]=(d/'generation_inputs.jsonl').read_bytes()
    paired=traces['baseline']==traces['caption_adapter'] and len(set(sources.values()))==1
    result={'status':'complete','paired_inputs_and_rng_verified':paired,'run_id':m['run_id'],
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'arms':{a:{k:summarize_series(v)for k,v in q.items()}for a,q in metrics.items()},
        'raw_metric_values':metrics,'paired_deltas':{},
        'limitations':['Available official InterHuman subset, not full original data.','Caption-only existing adapter; not full M1/M2/M3.','Rounded four-decimal upstream logs.','Sampling-repeat uncertainty, not independent retraining.','No independent contact or robot training utility claim.']}
    if paired:
        result['paired_deltas']={k:summarize_series([b-a for a,b in zip(metrics['baseline'][k],metrics['caption_adapter'][k])])for k in metrics['baseline']}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# BM1 生成评价结果','',f'两组输入与随机数配对核验：{paired}。',
           '差值为适配器−原始模型；FID和MM Distance越低越好，R-precision越高越好。多样性与Multimodality不能简单视为越高越好。',
           '','|指标|原始模型均值|适配器均值|配对差值|','|---|---:|---:|---:|']
    for k in metrics['baseline']:
        delta=result['paired_deltas'].get(k,{}).get('mean')
        lines.append('|'+k+'|'+f"{result['arms']['baseline'][k]['mean']:.4f}|{result['arms']['caption_adapter'][k]['mean']:.4f}|"+(f'{delta:.4f}'if delta is not None else'未通过配对核验')+'|')
    lines+=['','仅说明通用交互生成评价；不证明格斗接触正确、机器人可执行或合成数据提高模仿学习。完整数值与不确定性见 suite_result.json。']
    (out/'suite_result.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('BM1_SUITE_FINALIZED',flush=True)


if __name__=='__main__':main()
