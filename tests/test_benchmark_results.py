import unittest
from combat.benchmark_results import parse_metric_log, summarize_series

LOG='''---> [baseline] MM Distance: 3.0
---> [baseline] R_precision: (top 1): 0.1 (top 2): 0.2 (top 3): 0.3
---> [baseline] FID: 5.0
---> [baseline] Diversity: 8.0
---> [baseline] Multimodality: 1.0
========== FID Summary ==========
---> [baseline] Mean: 5.0 CInterval: 0.0
'''

class Results(unittest.TestCase):
    def test_summary_not_double_counted(self):
        self.assertEqual(parse_metric_log(LOG,'baseline',1)['FID'],[5.0])
    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError):parse_metric_log(LOG,'baseline',2)
    def test_sampling_uncertainty(self):
        r=summarize_series([1,3]);self.assertEqual(r['mean'],2)
        self.assertAlmostEqual(r['normal_95_halfwidth'],1.96)

if __name__=='__main__':unittest.main()
