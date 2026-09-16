import unittest
import numpy as np
from combat.remocap_data import canonicalize,common_resample
from combat.remocap_eval import TrainContactLibrary,metrics
from combat.data_refinement import refine_data


class RemocapProtocolTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(1)
        self.pair=rng.normal(size=(40,2,22,3)).astype(np.float32)

    def test_canonical_frame_depends_only_on_actor(self):
        changed=self.pair.copy();changed[:,1]+=100
        original=canonicalize(self.pair);new=canonicalize(changed)
        np.testing.assert_array_equal(original[:,0],new[:,0])

    def test_resample_uses_seconds_not_assumed_25fps(self):
        t=np.arange(101)/50
        pose=np.zeros((101,22,3));pose[:,:,0]=t[:,None]
        result=common_resample(t,pose,t,pose,20)
        self.assertEqual(result.shape,(41,2,22,3))
        np.testing.assert_allclose(result[:,0,0,0],np.arange(41)/20,atol=1e-7)

    def test_exact_reference_has_zero_errors(self):
        m=metrics(self.pair[:,0],self.pair[:,1],self.pair[:,1])
        for k in ['mpjpe_mm','velocity_error_mm_s','bone_length_error_mm','proximity_gap_mae_mm']:
            self.assertEqual(m[k],0)
        self.assertEqual(m['proximity_false_positive'],0)
        self.assertEqual(m['proximity_false_negative'],0)

    def test_retrieval_uses_training_actor_only(self):
        train=np.stack([self.pair,self.pair+3])
        library=TrainContactLibrary(train,[{'source':'train_a'},{'source':'train_b'}])
        _,info=library.propose(self.pair[:,0])
        self.assertEqual(info['training_source'],'train_a')
        self.assertFalse(info['test_reactor_used'])

    def test_temporal_can_run_without_contact_ground_truth(self):
        y,_=refine_data(self.pair,np.full(22,.05),20,None,0.,1e-5,steps=2)
        np.testing.assert_array_equal(y[:,0],self.pair[:,0])
        self.assertTrue(np.isfinite(y).all())

    def test_contact_rejects_missing_plan(self):
        with self.assertRaises(ValueError):
            refine_data(self.pair,np.full(22,.05),20,None,1.,0.,steps=2)


if __name__=='__main__':unittest.main()
