import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.summarize_tracking_diagnostics import check_evaluation, summarize, JOBS


def metric():
    return {'checkpoint_sha256': 'checkpoint', 'motion_sha256': 'reference', 'seed': 123,
            'environments': 32, 'held_out_motion': False,
            'strict_thresholds': {'mean_body_position_error_m': .15, 'mean_root_position_error_m': .15,
                                 'mean_joint_error_rad': .3, 'scope': 'declared research diagnostic thresholds, not original BeyondMimic success definition'},
            'initial_reference_alignment_cache_primed': True, 'observation_noise': False,
            'training_reset_and_dynamics_randomization': True, 'robots_per_environment': 1,
            'first_failure_steps': [100] * 32, 'horizon_steps': 147, 'control_dt': .02,
            'strict_tracking_pass_fraction': 0., 'completion_fraction': 0.,
            'mean_alive_error': {'joint_mae_rad': .2, 'root_error_m': .1, 'body_error_m': .12}}


class SummaryTest(unittest.TestCase):
    def test_wrong_provenance_refused(self):
        with self.assertRaises(ValueError):
            check_evaluation(metric(), 'different checkpoint', 'reference', False)

    def test_nonfinite_metric_refused(self):
        m = metric(); m['mean_alive_error']['root_error_m'] = float('nan')
        with self.assertRaises(ValueError):
            check_evaluation(m, 'checkpoint', 'reference', False)

    def test_changed_threshold_refused(self):
        m = metric(); m['strict_thresholds']['mean_body_position_error_m'] = .5
        with self.assertRaises(ValueError):
            check_evaluation(m, 'checkpoint', 'reference', False)

    def test_incomplete_evaluation_and_transfer_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(path, value):
                target = root / path; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value))
            refs = [{'clip_id': f'sample_{i:03}', 'sha256': f'train{i}'} for i in range(4)]
            refdir = 'runs/tracking_diagnostics_20260915_refs'
            write(refdir + '/protocol.json', {'jobs': list(JOBS), 'seed': 0, 'updates': 1000,
                  'envs': 512, 'steps_per_env': 24, 'references': refs})
            (root / refdir / 'single.npz').write_bytes(b'fixture bundle')
            write('runs/kungfu_test4_tracking_refs/manifest.json', {'records': [
                {'sample': f'sample_{i:03}', 'sha256': f'dev{i}'} for i in range(4)]})
            run = 'runs/tracking_diagnostic_single_uniform_seed0'
            write(run + '/config.json', {'mode': 'uniform', 'seed': 0, 'iterations': 1000,
                  'environments': 512, 'steps_per_env': 24, 'clip_ids': ['sample_000'],
                  'initialization': 'random; no checkpoint',
                  'bundle_sha256': hashlib.sha256(b'fixture bundle').hexdigest()})
            write(run + '/TRAINING_COMPLETE.json', {'environment_steps': 12288000, 'per_clip_steps': [12288000]})
            write('runs/tracking_diagnostic_smoke_single_uniform/SMOKE_COMPLETE.json', {'environment_steps': 9600})
            (root / run / 'model_999.pt').write_bytes(b'fixture checkpoint')
            for split, count in [('eval', 4), ('dev', 3)]:
                for i in range(count):
                    m = metric(); m['checkpoint_sha256'] = hashlib.sha256(b'fixture checkpoint').hexdigest()
                    m['motion_sha256'] = ('train' if split == 'eval' else 'dev') + str(i)
                    m['held_out_motion'] = split == 'dev'
                    write(f'runs/tracking_diagnostic_{split}_single_uniform_sample_{i:03}/metrics.json', m)
            result = summarize(root)
            self.assertEqual(result['complete_jobs'], 0)
            self.assertEqual(result['rows'][1]['status'], 'awaiting_evaluation')
            m = copy.deepcopy(m); m['motion_sha256'] = 'dev3'
            write('runs/tracking_diagnostic_dev_single_uniform_sample_003/metrics.json', m)
            row = summarize(root)['rows'][1]
            self.assertEqual(row['status'], 'complete')
            self.assertEqual(row['macro']['seen_training']['references'], 1)
            self.assertEqual(row['macro']['selected_transfer']['references'], 3)
            self.assertEqual(row['macro']['development']['references'], 4)


if __name__ == '__main__':
    unittest.main()
