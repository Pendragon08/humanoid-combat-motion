import unittest
import numpy as np
import torch
from combat.coverage_critic import reference_features,ControllerCoverageCritic,prompt_split
from combat.protocol import validate_physics_label

class Step7Tests(unittest.TestCase):
    def test_translation_keeps_pair_geometry(self):
        q=np.zeros((6,2,36),np.float32);q[:,:,3]=1;q[:,1,0]=2
        shifted=q.copy();shifted[:,:,:3]+=np.array([12,-3,6])
        np.testing.assert_allclose(reference_features(q,50),reference_features(shifted,50))
        self.assertEqual(reference_features(q,50)[0,68],2)

    def test_padding_does_not_change_score(self):
        torch.manual_seed(7);model=ControllerCoverageCritic().eval();x=torch.randn(1,10,136)
        a=model(x,torch.tensor([10]));b=model(torch.cat([x,torch.full((1,7,136),999.)],1),torch.tensor([10]))
        torch.testing.assert_close(a,b)

    def test_unapproved_full_labels_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unapproved'):validate_physics_label({'critic_training_labels_approved':False})
        with self.assertRaisesRegex(ValueError,'Contact censoring'):validate_physics_label({'critic_training_labels_approved':True})

    def test_prompt_groups_not_random_frames(self):
        self.assertEqual([prompt_split(i) for i in range(4)],['train','train','val','test'])
        with self.assertRaises(ValueError):prompt_split(4)
