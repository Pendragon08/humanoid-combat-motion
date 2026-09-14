import unittest
import numpy as np
from combat.identity import stabilize_actor_identity


class IdentityTests(unittest.TestCase):
    def test_recovers_framewise_swap_without_changing_geometry(self):
        x=np.zeros((9,2,3,3));x[:,1,:,0]=2;x[...,1]=np.arange(3)[None,None,:]
        x[:,:,:,2]=np.arange(9)[:,None,None]*.03
        corrupt=x.copy();corrupt[[2,3,7]]=corrupt[[2,3,7],::-1]
        fixed,idx,audit=stabilize_actor_identity(corrupt)
        np.testing.assert_array_equal(fixed,x)
        self.assertEqual(audit['frames_permuted'],3)
        self.assertLess(audit['corrected_continuity_cost'],audit['raw_continuity_cost'])
        for t in range(len(x)):
            self.assertEqual(set(idx[t]),{0,1})
    def test_continuous_motion_unchanged(self):
        x=np.zeros((5,2,2,3));x[:,1,:,0]=3
        np.testing.assert_array_equal(stabilize_actor_identity(x)[0],x)
    def test_rejects_nonfinite(self):
        x=np.zeros((5,2,2,3));x[1,0,0,0]=np.nan
        with self.assertRaises(ValueError):stabilize_actor_identity(x)
