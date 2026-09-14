import unittest
import torch
import numpy as np
from combat.contact_guidance import refine_contacts
from combat.contact_kinematics import refine_kinematic_contacts
from combat.interaction import ContactEvent
from combat.reaction import ActorMotionAdapter,CausalReactionResidual,masked_defender_mse,history_condition


class ReactionTests(unittest.TestCase):
    def test_cross_attention_masks_future_actor_keys(self):
        torch.manual_seed(7);net=CausalReactionResidual();torch.nn.init.normal_(net.out.weight,std=.01)
        x=torch.randn(1,8,524);actor=torch.randn(1,8,262);observed=torch.tensor([6])
        before=net(x,actor,observed)
        actor[:,3:]=1000;after=net(x,actor,observed)
        torch.testing.assert_close(before[:,:3],after[:,:3])
        actor=torch.randn(1,8,262);before=net(x,actor,observed);actor[:,6:]=1000
        torch.testing.assert_close(before,net(x,actor,observed))
    def test_differentiable_fk_preserves_bones_and_reduces_gap(self):
        x=np.zeros((6,2,3,3),dtype=np.float32);x[:,:,1,1]=.2;x[:,:,2,1]=.4;x[:,1,:,0]=.45
        y,_=refine_kinematic_contacts(x,np.full(3,.05),30,[ContactEvent('hit',0,5,2,2)],[(0,1),(1,2)],steps=60)
        np.testing.assert_array_equal(y[:,0],x[:,0])
        np.testing.assert_allclose(np.linalg.norm(np.diff(y,axis=2),axis=-1),.2,atol=1e-6)
        self.assertLess(np.linalg.norm(y[:,0,2]-y[:,1,2],axis=-1).min(),.2)
    def test_future_and_boundary_differences_do_not_enter_history(self):
        class Identity:
            def forward(self,x):return x
        x=torch.randn(1,10,262);cutoff=torch.tensor([5])
        first=history_condition(x,cutoff,Identity())
        x[:,5:]=10000;x[:,4,66:132]=10000;x[:,4,258:]=10000
        torch.testing.assert_close(first,history_condition(x,cutoff,Identity()))
    def test_future_loss_excludes_observed_prefix(self):
        x=torch.zeros(1,8,524);y=x.clone();y[:,:4,262:]=100
        self.assertEqual(masked_defender_mse(y,x,torch.tensor([8]),torch.tensor([4])).item(),0)
    def test_projected_refinement_keeps_bone_lengths(self):
        x=np.zeros((5,2,3,3),dtype=np.float32)
        x[:,:,1,1]=.2;x[:,:,2,1]=.4;x[:,1,:,0]=.4
        y,_=refine_contacts(x,np.full(3,.05),30,[ContactEvent('hit',0,4,2,2)],
                           [(0,1),(1,2)],steps=10,fixed_actor=0,hard_bone_projection=True)
        np.testing.assert_allclose(np.linalg.norm(y[:,:,1:]-y[:,:,:-1],axis=-1),.2,atol=1e-6)
        np.testing.assert_array_equal(y[:,0],x[:,0])
    def test_contact_refinement_preserves_supplied_actor(self):
        x=np.zeros((5,2,2,3),dtype=np.float32);x[:,1,:,0]=.4
        refined,_=refine_contacts(x,np.array([.05,.05]),30,[ContactEvent('hit',0,4,0,0)],
                                 [(0,1)],steps=20,lr=.01,fixed_actor=0)
        np.testing.assert_array_equal(refined[:,0],x[:,0])
        self.assertLess(np.linalg.norm(refined[:,1,0],axis=-1).mean(),.4)
    def test_padding_cannot_change_condition(self):
        torch.manual_seed(0);net=ActorMotionAdapter()
        torch.nn.init.normal_(net.out.weight,std=.01)
        x=torch.randn(1,8,262);text=torch.randn(1,768);length=torch.tensor([5])
        kind=torch.tensor([1]);window=torch.tensor([[.2,.8]])
        first=net(text,x,length,kind,window);x[:,5:]=1e5
        torch.testing.assert_close(first,net(text,x,length,kind,window))
    def test_loss_excludes_condition_actor_and_padding(self):
        target=torch.zeros(1,5,524);pred=torch.zeros_like(target)
        pred[:,:,:262]=100;pred[:,3:,262:]=100
        self.assertEqual(masked_defender_mse(pred,target,torch.tensor([3])).item(),0)
        pred[:,:3,262:]=2
        self.assertEqual(masked_defender_mse(pred,target,torch.tensor([3])).item(),4)


if __name__=='__main__':unittest.main()
