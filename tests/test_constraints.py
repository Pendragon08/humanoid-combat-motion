"""Meaningful invariants: projection bounds, quaternion handling, recorded controller limits."""
import unittest
import numpy as np
import json
from combat.robot import Robot
from combat.constraints import project_joints

class ConstraintsTest(unittest.TestCase):
    def test_projection_violations(self):
        r=Robot();rng=np.random.default_rng(42)
        q=np.zeros((32,36));q[:,2]=.8;q[:,3]=1
        q[:,7:]=rng.normal(size=(32,29))*2
        out,_=project_joints(q,r)
        self.assertLessEqual(np.max(out[:,7:]-r.limits[:,1]),2e-4)
        self.assertLessEqual(np.max(r.limits[:,0]-out[:,7:]),2e-4)
        self.assertLessEqual(np.max(np.abs(np.diff(out[:,7:],axis=0)*30)-r.velocity_limits),2e-3)
        self.assertLessEqual(np.max(np.abs(np.diff(out[:,7:],n=2,axis=0)*900)-r.acceleration_limits),.2)
        np.testing.assert_array_equal(q[:,:7],out[:,:7])
        self.assertTrue(np.all(r.torque_limits>0))

    def test_quaternion_and_joint_order(self):
        r=Robot();q=np.zeros((3,36));q[:,3]=[1,-1,1]
        q[:,7:]=np.arange(29)
        out=r.remap(q)
        np.testing.assert_allclose(np.linalg.norm(out[:,3:7],axis=1),1)
        self.assertTrue(np.all(np.sum(out[1:,3:7]*out[:-1,3:7],axis=1)>0))
        np.testing.assert_array_equal(out[0,7:],np.arange(29)[r.mapping])

    def test_physics_diagnostic_serializable(self):
        r=Robot();q=np.tile(r.model.qpos0,(8,1))
        result,poses=r.diagnostic_rollout(q,record=True)
        json.dumps(result)
        self.assertEqual(poses.shape[1],36)
        self.assertIsInstance(result['pd_full_horizon_success'],bool)
        self.assertGreater(result['duration_s'],0)

if __name__=='__main__':unittest.main()
