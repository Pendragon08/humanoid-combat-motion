"""Summarize all final cross-target evaluations and enforce equal exposure/init."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_matched_data_20260917'
OUT=ROOT/'runs/pbhc_matched_data_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target=REC/'summary.json'
    if target.exists():raise FileExistsError(target)
    manifest=json.loads((REC/'manifest.json').read_text());cfg=manifest['config'];rows=[];training=[]
    for job in manifest['jobs']:
        folder=ROOT/job['training_output'];name=job['name'];arm=job['arm']
        initial=json.loads((folder/'initialization.json').read_text())
        complete=json.loads((folder/'TRAINING_COMPLETE.json').read_text())
        receipt=json.loads((folder/(cfg['final_checkpoint']+'.receipt.json')).read_text())
        assert complete['environment_steps']==receipt['environment_steps']==cfg['iterations']*cfg['num_envs']*cfg['steps_per_env']
        original=json.loads((OUT/'training'/f'{name}_raw'/'initialization.json').read_text())
        for k in [k for k in initial if k.endswith('_sha256')]:assert initial[k]==original[k]
        training.append(dict(name=name,arm=arm,initialization=initial,receipt=receipt,checkpoint_sha256=sha(folder/cfg['final_checkpoint'])))
        record=REC/'training'/f'{name}_{arm}';record.mkdir(parents=True)
        for file in ['initialization.json','TRAINING_COMPLETE.json','config.json','env.yaml','agent.yaml',cfg['final_checkpoint']+'.receipt.json']:
            shutil.copy2(folder/file,record/file)
        for reference in cfg['evaluation_references']:
            for regime in cfg['evaluation_regimes']:
                key=f'{name}_{arm}_on_{reference}_{regime}';path=OUT/'evaluation'/key/'metrics.json';m=json.loads(path.read_text())
                assert m['checkpoint_sha256']==training[-1]['checkpoint_sha256'] and m['reference_sha256']==job['references'][reference]['sha256']
                rows.append(dict(name=name,train_arm=arm,evaluation_reference=reference,regime=regime,
                    completion=m['completion_fraction'],strict=m['strict_every_frame_pass_fraction'],root_mean_m=m['mean_active_errors'][2],metrics_sha256=sha(path)))
                dest=REC/'evaluation'/key;dest.mkdir(parents=True);shutil.copy2(path,dest/'metrics.json')
    assert len(rows)==16 and len(training)==4
    output=dict(scope=cfg['scope'],training=training,results=rows,primary_endpoint=cfg['primary_endpoint'],
        caveats=['Two development clips; single training seed; not independent test or final data utility',
                 'Compare raw and quality models on the SAME evaluation reference and regime; own-reference success alone is insufficient',
                 'Different initial trajectories can still change environment random draws; matching weights is not identical optimization experience',
                 'Early termination censors mean errors; always interpret with completion and strict success'])
    target.write_text(json.dumps(output,indent=2))
    lines=['# 等预算数据训练对照','',cfg['scope'],'',
        '两动作×两数据组；每组512环境×24步×1000更新，随机初始化种子0，参数初值逐张量哈希匹配；固定最终model_999，不挑检查点。','',
        '|动作|训练数据|评估参考|条件|完整执行|严格通过|根误差(m)|','|---|---|---|---|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["name"]}|{r["train_arm"]}|{r["evaluation_reference"]}|{r["regime"]}|{r["completion"]:.2%}|{r["strict"]:.2%}|{r["root_mean_m"]:.4f}|')
    lines+=['','主比较为同一未修改原始参考、同一随机扰动条件下的严格通过率。均值包含首次终止帧，必须结合完成率解释。没有独立动作测试、多种子或双人格斗收益结论。']
    (REC/'report.md').write_text('\n'.join(lines)+'\n')
    # Preserve all raw evaluation arrays, small receipts and logs; weights remain on server.
    archive=OUT/'evaluation_archive.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(OUT/'evaluation',arcname='evaluation');tar.add(REC,arcname='records')
    (REC/'archive.json').write_text(json.dumps(dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size),indent=2))
    print('MATCHED_DATA_COMPLETE',len(rows))

if __name__=='__main__':main()
