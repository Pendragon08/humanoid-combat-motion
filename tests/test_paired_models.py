import unittest
import torch
from combat.paired_models import CombatConditionAdapter,DualRobotPhysicsCritic,critic_loss,rank_candidates
from combat.contact_guidance import contact_objective
from combat.interaction import ContactEvent


class PairedModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(0)

    def test_adapter_preserves_pretrained_condition_at_init(self):
        model=CombatConditionAdapter({'attack':4,'defense':4},dim=16)
        text=torch.randn(2,16)
        cond={'attack':torch.tensor([1,2]),'defense':torch.tensor([2,1])}
        torch.testing.assert_close(model(text,cond),text)
        model(text,cond).square().mean().backward()
        self.assertGreater(model.out.weight.grad.abs().sum(),0)

    def test_critic_finite_backward_and_padding(self):
        model=DualRobotPhysicsCritic(12)
        x=torch.randn(2,10,12); lengths=torch.tensor([10,6])
        out=model(x,lengths)
        x[1,6:]=1e6
        other=model(x,lengths)
        for key in out:torch.testing.assert_close(out[key],other[key])
        target={'tracking_error_m':torch.ones(2)*.05,'fall':torch.tensor([0,1]),'contact_success':torch.tensor([1,0]),'invalid_collision':torch.tensor([0,1])}
        loss=critic_loss(out,target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.head.weight.grad.abs().sum(),0)
        order,score=rank_candidates(out)
        self.assertEqual(sorted(order.tolist()),[0,1])

    def test_contact_gradient_closes_miss(self):
        x=torch.zeros(5,2,2,3);x[:,1,:,0]=.4;x.requires_grad_(True)
        loss=contact_objective(x,torch.tensor([.05,.05]),[ContactEvent('hit',0,4,0,0)])
        loss.backward()
        self.assertLess(x.grad[:,0,0,0].sum(),0)
        self.assertGreater(x.grad[:,1,0,0].sum(),0)

    def test_critic_independent_of_batch_padding_length(self):
        model=DualRobotPhysicsCritic(12).eval()
        x=torch.randn(1,6,12)
        lengths=torch.tensor([6])
        short=model(x,lengths)
        padded=model(torch.cat([x,torch.randn(1,9,12)],dim=1),lengths)
        for key in short:torch.testing.assert_close(short[key],padded[key],atol=1e-6,rtol=1e-5)

    def test_avoid_gradient_increases_clearance(self):
        x=torch.zeros(5,2,2,3);x[:,1,:,0]=.15;x.requires_grad_(True)
        loss=contact_objective(x,torch.tensor([.05,.05]),[ContactEvent('avoid',0,4,0,0,threatened=True)])
        loss.backward()
        self.assertGreater(x.grad[:,0,0,0].sum(),0)
        self.assertLess(x.grad[:,1,0,0].sum(),0)


if __name__=='__main__':unittest.main()
