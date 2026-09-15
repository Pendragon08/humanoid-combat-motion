import unittest
import numpy as np
from combat.data_refinement import EDGES, refine_data, quality_metrics
from combat.interaction import ContactEvent

class DataRefinementTests(unittest.TestCase):
    def motion(self):
        x=np.zeros((12,2,22,3),dtype=np.float32)
        for actor in (0,1):
            x[:,actor,0]=[actor*.6,1,0]
            for p,c in EDGES:
                x[:,actor,c]=x[:,actor,p]+np.array([.08,.04,.03])
        x[:,1,:,0]+=np.sin(np.arange(12)[:,None])*.03
        return x,np.full(22,.055),ContactEvent('hit',2,8,20,15,confidence=.4,provenance='unit_test')

    def test_contact_and_temporal_preserve_attacker_and_source_bones(self):
        x,r,e=self.motion()
        y,_=refine_data(x,r,30,e,1,1e-5,steps=5)
        m=quality_metrics(y,x,r,30,e)
        self.assertLess(m['max_bone_length_change_m'],1e-6)
        self.assertEqual(m['max_attacker_change_m'],0)
        self.assertFalse(m['robot_dynamics_verified'])

    def test_raw_identity_has_zero_change_and_unit_path_ratio(self):
        x,r,e=self.motion()
        m=quality_metrics(x,x,r,30,e)
        self.assertEqual(m['defender_displacement_rms_m'],0)
        self.assertAlmostEqual(m['defender_path_length_ratio'],1)

if __name__=='__main__':unittest.main()
