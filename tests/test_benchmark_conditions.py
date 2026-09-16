import unittest
from combat.benchmark_conditions import caption_conditions, assert_no_test_overlap


class BenchmarkConditions(unittest.TestCase):
    def test_only_caption_information(self):
        self.assertEqual(caption_conditions('They sit and greet.'),
                         dict(punch=0, kick=0, boxing=0, martial_arts=0, defense=0))
        tags = caption_conditions('A boxer punches; another dodges and blocks.')
        self.assertEqual(tags, dict(punch=1, kick=0, boxing=1, martial_arts=0, defense=1))

    def test_word_boundaries(self):
        self.assertEqual(caption_conditions('a mailbox and kickstand')['kick'], 0)
        self.assertEqual(caption_conditions('a mailbox and kickstand')['boxing'], 0)

    def test_selection_leakage_rejected(self):
        p = dict(training_sources=[['interhuman:1']], validation_sources=[['interhuman:2']])
        assert_no_test_overlap(p, ['3', '4'])
        with self.assertRaises(ValueError):
            assert_no_test_overlap(p, ['2'])


if __name__ == '__main__':
    unittest.main()
