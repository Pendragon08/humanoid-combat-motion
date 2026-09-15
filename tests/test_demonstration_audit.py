import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_demonstration_sources import duplicate_sets, inspect_motion


class DemonstrationAuditTests(unittest.TestCase):
    def write_motion(self, folder, fps=50, q=None):
        if q is None:
            q=np.zeros((6,36)); q[:,3]=1
            q[:,7]=np.arange(6)*0.1
        path=Path(folder)/'motion.npz'
        np.savez(path,qpos=q,fps=fps)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_velocity_uses_recorded_fps_and_duration_uses_intervals(self):
        with tempfile.TemporaryDirectory() as folder:
            path,h=self.write_motion(folder)
            audit=inspect_motion(path,h)
            self.assertTrue(audit['structural_valid'])
            self.assertAlmostEqual(audit['joint_speed_abs_max_rad_s'],5)
            self.assertAlmostEqual(audit['duration_s'],0.1)
            self.assertFalse(audit['dynamics_verified'])

    def test_hash_mismatch_is_not_accepted_as_valid_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=self.write_motion(folder)
            audit=inspect_motion(path,'0'*64)
            self.assertFalse(audit['structural_valid'])
            self.assertIn('source_hash_mismatch',audit['errors'])

    def test_invalid_timebase_and_quaternion_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path,h=self.write_motion(folder,fps=0)
            self.assertFalse(inspect_motion(path,h)['structural_valid'])
            path,h=self.write_motion(folder,q=np.zeros((6,36)))
            audit=inspect_motion(path,h)
            self.assertFalse(audit['structural_valid'])
            self.assertIn('zero_root_quaternion',audit['errors'])

    def test_numeric_duplicates_flag_cross_split_leakage(self):
        rows=[{'sample_id':'a','split':'train','quality':{'numeric_motion_sha256':'same'}},
              {'sample_id':'b','split':'test','quality':{'numeric_motion_sha256':'same'}}]
        result=duplicate_sets(rows)
        self.assertEqual(len(result),1)
        self.assertTrue(result[0]['cross_split'])

    def test_readable_motion_is_distinct_from_ready_to_export(self):
        with tempfile.TemporaryDirectory() as folder:
            q=np.zeros((6,36)); q[:,3]=1.05
            path,h=self.write_motion(folder,q=q)
            result=inspect_motion(path,h)
            self.assertTrue(result['structural_valid'])
            self.assertFalse(result['raw_reference_ready'])
            self.assertIn('requires_root_quaternion_normalization', result['warnings'])


if __name__=='__main__':
    unittest.main()
