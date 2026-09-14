import random
import unittest
import numpy as np
import torch
from scripts.train_timotion_adapter import fixed_validation_rng


class ValidationRngTest(unittest.TestCase):
    def test_fixed_draws_and_training_stream_restoration(self):
        np.random.seed(23);random.seed(23);torch.manual_seed(23)
        expected=(np.random.random(),random.random(),torch.rand(1).item())
        np.random.seed(23);random.seed(23);torch.manual_seed(23)
        def sample():return np.random.random(),random.random(),torch.rand(1).item()
        with fixed_validation_rng():first=sample()
        with fixed_validation_rng():second=sample()
        self.assertEqual(first,second)
        self.assertEqual(sample(),expected)


if __name__=='__main__':unittest.main()
