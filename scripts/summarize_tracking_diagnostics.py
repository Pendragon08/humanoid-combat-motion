"""Audit the frozen four-job protocol and summarize actual, complete evaluations.

Never infer evaluation success from a process exit or training reward. Uses only
the standard library, so the validated simulator environment stays unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics

JOBS = ('single_upstream', 'single_uniform', 'four_uniform', 'four_adaptive')
KEYS = ('strict', 'completion', 'coverage_s', 'joint_error_rad', 'root_error_m', 'body_error_m')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_evaluation(m, checkpoint_sha, reference_sha, held_out):
    if m['checkpoint_sha256'] != checkpoint_sha or m['motion_sha256'] != reference_sha:
        raise ValueError('Evaluation checkpoint/reference mismatch')
    if (m['seed'], m['environments'], m['held_out_motion']) != (123, 32, held_out):
        raise ValueError('Evaluation protocol mismatch')
    if m['strict_thresholds'] != {'mean_body_position_error_m': .15,
            'mean_root_position_error_m': .15, 'mean_joint_error_rad': .3,
            'scope': 'declared research diagnostic thresholds, not original BeyondMimic success definition'}:
        raise ValueError('Evaluation thresholds changed')
    if not m['initial_reference_alignment_cache_primed'] or m['observation_noise']:
        raise ValueError('Evaluation alignment/noise changed')
    if not m['training_reset_and_dynamics_randomization'] or m['robots_per_environment'] != 1:
        raise ValueError('Evaluation randomization/robot count changed')
    if len(m['first_failure_steps']) != 32 or not all(0 <= t <= m['horizon_steps'] for t in m['first_failure_steps']):
        raise ValueError('Invalid coverage records')
    if not 0 <= m['strict_tracking_pass_fraction'] <= m['completion_fraction'] <= 1:
        raise ValueError('Invalid success fractions')
    if not all(math.isfinite(float(v)) for v in m['mean_alive_error'].values()):
        raise ValueError('Nonfinite metrics')
    return {'strict': m['strict_tracking_pass_fraction'], 'completion': m['completion_fraction'],
            'coverage_s': statistics.mean(m['first_failure_steps']) * m['control_dt'],
            'joint_error_rad': m['mean_alive_error']['joint_mae_rad'],
            'root_error_m': m['mean_alive_error']['root_error_m'],
            'body_error_m': m['mean_alive_error']['body_error_m']}


def summarize(root):
    refdir = root / 'runs/tracking_diagnostics_20260915_refs'
    protocol = read(refdir / 'protocol.json')
    if (protocol['jobs'], protocol['seed'], protocol['updates'], protocol['envs'], protocol['steps_per_env']) != (list(JOBS), 0, 1000, 512, 24):
        raise ValueError('Frozen protocol changed')
    selected = protocol['references']
    development = read(root / 'runs/kungfu_test4_tracking_refs/manifest.json')['records']
    if len(selected) != 4 or len(development) != 4:
        raise ValueError('Unexpected reference count')
    rows = []
    for job in JOBS:
        run = root / f'runs/tracking_diagnostic_{job}_seed0'
        mode = job.split('_', 1)[1]
        single = job.startswith('single_')
        ids = [x['clip_id'] for x in selected[:1 if single else 4]]
        row = {'job': job, 'status': 'queued', 'evaluations': [], 'training_clip_ids': ids}
        smoke = root / f'runs/tracking_diagnostic_smoke_{job}/SMOKE_COMPLETE.json'
        row['smoke_passed'] = smoke.exists() and read(smoke)['environment_steps'] == 9600
        if (run / 'config.json').exists():
            cfg = read(run / 'config.json')
            if (cfg['mode'], cfg['seed'], cfg['iterations'], cfg['environments'], cfg['steps_per_env'], cfg['clip_ids']) != (mode, 0, 1000, 512, 24, ids):
                raise ValueError('Training budget/data/config mismatch: ' + job)
            if cfg['initialization'] != 'random; no checkpoint' or cfg['bundle_sha256'] != sha(refdir / ('single.npz' if single else 'four.npz')):
                raise ValueError('Initialization/bundle mismatch: ' + job)
            row.update(status='training_or_incomplete', config_sha256=sha(run / 'config.json'))
            log = root / f'records/tracking_diagnostics_20260915/train_{job}.log'
            if log.exists():
                steps = re.findall(r'Total timesteps:\s*(\d+)', log.read_text(errors='replace'))
                row['last_logged_steps'] = int(steps[-1]) if steps else 0
        marker = run / 'TRAINING_COMPLETE.json'
        checkpoint = run / 'model_999.pt'
        if marker.exists():
            if not checkpoint.exists() or not row['smoke_passed']:
                raise ValueError('Missing checkpoint or successful smoke: ' + job)
            exposure = read(marker)
            if exposure['environment_steps'] != 12288000 or sum(exposure['per_clip_steps']) != 12288000 or len(exposure['per_clip_steps']) != len(ids):
                raise ValueError('Actual training exposure mismatch: ' + job)
            row.update(status='awaiting_evaluation', exposure=exposure, model_sha256=sha(checkpoint))
            refs = [('eval', x['clip_id'], x['sha256'], False) for x in selected]
            refs += [('dev', x['sample'], x['sha256'], True) for x in development]
            for split, sample, expected_sha, held in refs:
                path = root / f'runs/tracking_diagnostic_{split}_{job}_{sample}/metrics.json'
                if not path.exists():
                    continue
                values = check_evaluation(read(path), row['model_sha256'], expected_sha, held)
                values.update(sample=sample, split=split, seen_in_training=not held and sample in ids,
                              path=str(path.relative_to(root)), sha256=sha(path))
                row['evaluations'].append(values)
            if len(row['evaluations']) == 8:
                row['status'] = 'complete'
                groups = {'shared_first': lambda e: e['split'] == 'eval' and e['sample'] == selected[0]['clip_id'],
                          'selected_four': lambda e: e['split'] == 'eval',
                          'seen_training': lambda e: e['seen_in_training'],
                          'selected_transfer': lambda e: e['split'] == 'eval' and not e['seen_in_training'],
                          'development': lambda e: e['split'] == 'dev'}
                row['macro'] = {}
                for name, predicate in groups.items():
                    members = [e for e in row['evaluations'] if predicate(e)]
                    if members:
                        row['macro'][name] = dict(references=len(members), **{k: statistics.mean(e[k] for e in members) for k in KEYS})
        rows.append(row)
    differences = []
    lookup = {r['job']: r for r in rows}
    for treatment, control, comparison in [('single_uniform', 'single_upstream', 'interface package; reset/alignment/end handling also differ'),
                                          ('four_adaptive', 'four_uniform', 'bounded clip failure sampling; same four references')]:
        t, c = lookup[treatment], lookup[control]
        if t['status'] == c['status'] == 'complete':
            differences.append({'treatment': treatment, 'control': control, 'comparison': comparison,
                                'macro_deltas': {g: {k: t['macro'][g][k] - c['macro'][g][k] for k in KEYS}
                                                 for g in ('shared_first', 'selected_four', 'development')}})
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'protocol_sha256': sha(refdir / 'protocol.json'), 'rows': rows, 'differences': differences,
            'complete_jobs': sum(r['status'] == 'complete' for r in rows), 'expected_jobs': 4,
            'expected_evaluations': 32, 'evaluation_count': sum(len(r['evaluations']) for r in rows),
            'limitations': ['One training seed per condition; no seed-level uncertainty estimate.',
                           'Four development references share one source group and have been used previously.',
                           'Single-reference transfer to the other three selected references is not training-set performance.',
                           'Errors are censored at first termination; compare coverage and strict success together.',
                           'Tracking termination is not a separately measured fall/contact failure.',
                           'No generated motion is added in these four diagnostics; no synthetic-data utility claim.',
                           'References are not manually certified low-dynamic combat classes.']}


def update_html(path, result):
    start, end = '<!-- TRACKING-DIAGNOSTICS-RESULTS-BEGIN -->', '<!-- TRACKING-DIAGNOSTICS-RESULTS-END -->'
    section = start + '<section id="tracking-diagnostics-results"><h2>优化诊断：接口与采样方法的实际结果</h2>'
    section += f'<p>完整完成 {result["complete_jobs"]}/4 项，评价 {result["evaluation_count"]}/32 组。每项训练预算相同：512环境、1000次更新、12,288,000环境步，种子0。下表是每条参考先独立计算、再等权平均。</p>'
    section += '<table><tr><th>实验</th><th>状态</th><th>共同第一条：严格通过</th><th>四条选定参考：严格通过</th><th>开发留出：严格通过</th><th>开发留出：观察覆盖秒数</th></tr>'
    labels = {'single_upstream': '单参考／上游接口', 'single_uniform': '单参考／现有均匀接口',
              'four_uniform': '四参考／均匀采样', 'four_adaptive': '四参考／自适应采样'}
    states = {'queued': '排队中', 'training_or_incomplete': '训练中或尚未完整结束',
              'awaiting_evaluation': '训练完成，等待完整评价', 'complete': '训练和评价完成'}
    for row in result['rows']:
        section += '<tr><td>' + labels[row['job']] + '</td><td>' + states[row['status']] + '</td>'
        for group in ('shared_first', 'selected_four', 'development'):
            section += '<td>' + ('%.2f%%' % (100 * row['macro'][group]['strict']) if 'macro' in row else '待完整评价') + '</td>'
        section += '<td>' + ('%.3f' % row['macro']['development']['coverage_s'] if 'macro' in row else '—') + '</td></tr>'
    section += '</table><p>严格通过：全段未触发失配，且平均身体位置误差&lt;0.15米、根部位置误差&lt;0.15米、关节角误差&lt;0.3弧度。覆盖秒数是首次失配或全段完成前的时长；不是单独标注的跌倒时间。</p>'
    section += '<p class="notice">每个条件只有一个训练种子；开发留出来自同一个视频来源组，不能据此宣称统计显著或普遍泛化。单参考实验只有第一条是训练动作，其余三条是迁移检查。上游和现有接口还改变了重置、对齐及参考结束处理，差异不能全部归因于采样。这批实验未添加合成数据，不能证明合成数据效用。</p>'
    if result['complete_jobs'] == 4:
        rows = {r['job']: r for r in result['rows']}
        uniform, adaptive = rows['four_uniform'], rows['four_adaptive']
        dev_delta = 100 * (adaptive['macro']['development']['strict'] - uniform['macro']['development']['strict'])
        train_delta = 100 * (adaptive['macro']['selected_four']['strict'] - uniform['macro']['selected_four']['strict'])
        section += '<h3>每条训练参考的差异</h3><table><tr><th>参考编号</th><th>均匀采样严格通过</th><th>自适应严格通过</th><th>均匀：实际训练步数占比</th><th>自适应：实际训练步数占比</th></tr>'
        by_sample = {e['sample']: e for e in adaptive['evaluations'] if e['split'] == 'eval'}
        for i, e in enumerate(x for x in uniform['evaluations'] if x['split'] == 'eval'):
            section += '<tr><td>' + html.escape(e['sample']) + '</td><td>%.2f%%</td><td>%.2f%%</td><td>%.2f%%</td><td>%.2f%%</td></tr>' % (100*e['strict'], 100*by_sample[e['sample']]['strict'],
                100*uniform['exposure']['per_clip_steps'][i]/12288000, 100*adaptive['exposure']['per_clip_steps'][i]/12288000)
        section += '</table><p>均匀采样让四条动作被抽到的次数接近相同，但困难动作提前失配，实际交互步数仍明显较少。自适应提高了部分困难动作的训练占比；这不等于已解决完整动作执行问题。</p>'
        section += '<h3>本轮结论与下一步</h3><p>相较四参考均匀采样，自适应的开发严格通过率变化为%+.2f个百分点，训练参考平均变化为%+.2f个百分点。结果有动作依赖性，不能由单个种子确认稳定收益。</p>' % (dev_delta, train_delta)
        section += '<p class="notice"><b>本轮尚未建立足够可靠的基础跟踪能力，暂不扩大合成数据效用实验。</b>先用人工确认的站立、慢速摆臂、重心转移等简单参考做校准，再逐项检查初始状态对齐、根部速度、足底接触、动力学可执行性与训练时随机起点／评价时固定起点的差异。这些是下一轮待验证假设，不是已经确认的唯一失败原因。原方案的简单训练参考≥80%、匹配验证来源≥60%为建议工程门槛；本批参考尚未人工认证为简单基准，不能声称已完成该校准。</p>'
    all_video = 'tracking_diagnostic_sampling_all'
    if (path.parent / 'media' / (all_video + '.mp4')).exists():
        section += '<h3>完整展示四条选定训练参考</h3><video controls preload="metadata" poster="media/' + all_video + '.png"><source src="media/' + all_video + '.mp4" type="video/mp4"></video><p>按冻结清单展示全部四条参考；从上到下为000、008、010、014，每排左侧是参考，中间是均匀采样策略，右侧是自适应策略。固定环境0，未按成功率挑选展示动作。</p>'
    for comparison, title in [('interface', '单参考接口对比'), ('sampling', '四参考采样对比')]:
        stem = 'tracking_diagnostic_' + comparison
        if (path.parent / 'media' / (stem + '.mp4')).exists():
            section += '<h3>' + title + '</h3><video controls preload="metadata" poster="media/' + stem + '.png"><source src="media/' + stem + '.mp4" type="video/mp4"></video>'
    section += '<p>视频使用共同第一条训练参考和第一条开发留出参考，固定环境0，上排训练参考、下排开发参考；每排从左到右为参考与两种策略。首次失配后冻结，4倍慢放。MuJoCo仅渲染已记录的Isaac轨迹，不是跨仿真器策略测试。</p>'
    section += '<p><a href="../records/tracking_diagnostics_20260915/summary.json">详细结果、实际采样次数与来源校验</a></p></section>' + end
    source = path.read_text(encoding='utf-8')
    if start in source:
        source = source[:source.index(start)] + section + source[source.index(end) + len(end):]
    else:
        source = source.replace('<!-- STEP7-BEGIN -->', section + '\n<!-- STEP7-BEGIN -->', 1)
    path.write_text(source, encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    p.add_argument('--html', type=Path)
    a = p.parse_args()
    result = summarize(a.root)
    output = a.output or a.root / 'records/tracking_diagnostics_20260915/summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if a.html:
        update_html(a.html, result)
    print('TRACKING_DIAGNOSTICS_SUMMARY', result['complete_jobs'], '/4', result['evaluation_count'], '/32')


if __name__ == '__main__':
    main()
