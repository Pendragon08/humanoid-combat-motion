import unittest
import numpy as np
from combat.interaction import ContactEvent, evaluate_event, paired_features
from combat.protocol import source_split_check, validate_physics_label


class InteractionTests(unittest.TestCase):
    def motion(self, distance):
        x = np.zeros((5,2,3,3))
        x[:,1,:,0] = 5
        x[:,1,0,0] = distance
        return x, np.array([.05,.05,.05])

    def event(self,kind='hit',**kw):
        return ContactEvent(kind,0,4,0,0,**kw)

    def test_hit(self):
        x,r=self.motion([.5,.3,.12,.3,.5])
        out=evaluate_event(x,r,30,self.event())
        self.assertTrue(out['success'])
        self.assertEqual(out['first_contact_frame'],2)

    def test_miss(self):
        self.assertFalse(evaluate_event(*self.motion(.5),30,self.event())['success'])

    def test_penetration_is_not_hit(self):
        out=evaluate_event(*self.motion(.02),30,self.event())
        self.assertTrue(out['penetration'])
        self.assertFalse(out['success'])

    def test_tunnelling_detected(self):
        x,r=self.motion([.5,.5,-.5,-.5,-.5])
        self.assertTrue(evaluate_event(x,r,30,self.event())['penetration'])

    def test_block_then_protected_hit_is_failure(self):
        x,r=self.motion([.5,.12,.5,.5,.5])
        self.assertTrue(evaluate_event(x,r,30,self.event('block',protected_joint=1))['success'])
        x[3,1,1,0]=.12
        self.assertFalse(evaluate_event(x,r,30,self.event('block',protected_joint=1))['success'])

    def test_dodge_needs_threat(self):
        x,r=self.motion(.5)
        self.assertFalse(evaluate_event(x,r,30,self.event('avoid'))['success'])
        self.assertTrue(evaluate_event(x,r,30,self.event('avoid',threatened=True))['success'])
        x[:,1,0,0]=.15
        self.assertFalse(evaluate_event(x,r,30,self.event('avoid',threatened=True))['success'])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            evaluate_event(*self.motion(.5),30,ContactEvent('hit',0,9,0,0))

    def test_common_translation_invariance(self):
        x,r=self.motion([.5,.3,.12,.3,.5])
        np.testing.assert_allclose(paired_features(x,30),paired_features(x+2,30),atol=1e-5)
        a=evaluate_event(x,r,30,self.event())
        b=evaluate_event(x+2,r,30,self.event())
        self.assertEqual(a['success'],b['success'])

    def test_split_leakage(self):
        records=[dict(sample_id='a',source_ids=['source1'],split='train'),
                 dict(sample_id='b',source_ids=['source1'],split='test')]
        with self.assertRaises(ValueError): source_split_check(records)

    def test_test_condition_leakage(self):
        records=[dict(sample_id='a',source_ids=['s'],split='train',synthetic=True,conditioning_split='test')]
        with self.assertRaises(ValueError): source_split_check(records)

    def test_missing_physics_labels_fail(self):
        with self.assertRaises(ValueError): validate_physics_label({'controller_kind':'PD'})


if __name__=='__main__': unittest.main()
