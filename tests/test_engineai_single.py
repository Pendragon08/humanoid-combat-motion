import unittest
import numpy as np
from scripts.prepare_engineai_single import joint_project

class ProjectionTests(unittest.TestCase):
    def test_violation_correction_preserves_time_and_shape(self):
        q=np.zeros((30,2));q[10,0]=3.;q[:,1]=np.linspace(-2,2,30)
        bounds=np.array([[-1,1],[-.5,.5]]);v=np.array([2.,1.]);out=joint_project(q,bounds,v,30.,100.)
        self.assertEqual(out.shape,q.shape)
        self.assertTrue(np.all(out>=bounds[:,0]-1e-4));self.assertTrue(np.all(out<=bounds[:,1]+1e-4))
        self.assertTrue(np.all(np.abs(np.diff(out,axis=0))*30<=v+1e-3))
        self.assertLess(np.max(np.abs(np.diff(out,n=2,axis=0))*900),100.1)
    def test_constant_valid_pose_unchanged(self):
        q=np.full((20,2),.2);out=joint_project(q,np.array([[-1,1],[-1,1]]),np.array([2.,2.]),30.)
        np.testing.assert_allclose(out,q,atol=1e-5)
    def test_constraints_applied_after_rate_conversion(self):
        q=np.zeros((30,1));q[10:15]=1.
        bounds=np.array([[-2.,2.]]);v=np.array([35.2])
        projected=joint_project(q,bounds,v,30.)
        resampled=np.interp(np.arange(0,29/30,.02),np.arange(30)/30,projected[:,0])[:,None]
        self.assertGreater(np.max(np.abs(np.diff(resampled,n=2,axis=0))*2500),100.1)
        final=joint_project(resampled,bounds,v,50.)
        self.assertLess(np.max(np.abs(np.diff(final,n=2,axis=0))*2500),100.1)

if __name__=='__main__':unittest.main()
