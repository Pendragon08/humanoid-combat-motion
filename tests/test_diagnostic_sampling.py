import unittest
import torch
from combat.diagnostic_sampling import bounded_probabilities,sample_frames

class SamplingTest(unittest.TestCase):
    def test_bounded_failure_weight(self):
        p=bounded_probabilities(torch.tensor([0.,1.,.5,.1]))
        self.assertAlmostEqual(float(p.sum()),1.,places=6)
        self.assertLess(float(p.max()/p.min()),3.)
        self.assertGreater(float(p.min()),.2/4)
        self.assertGreater(float(p[1]),float(p[0]))
    def test_uniform_when_equal(self):
        torch.testing.assert_close(bounded_probabilities(torch.ones(4)),torch.full((4,),.25))
    def test_no_boundary_crossing(self):
        torch.manual_seed(0);starts=torch.tensor([0,100,150]);ends=torch.tensor([100,150,154])
        clips=torch.randint(3,(30000,));frames=sample_frames(starts,ends,clips)
        self.assertTrue(bool(torch.all(frames>=starts[clips])))
        self.assertTrue(bool(torch.all(frames<ends[clips]-2)))
    def test_nonfinite_refused(self):
        with self.assertRaises(ValueError):bounded_probabilities(torch.tensor([float('nan')]))

if __name__=='__main__':unittest.main()
