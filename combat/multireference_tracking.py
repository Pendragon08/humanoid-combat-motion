"""Uniform clip/frame sampling with explicit episode boundaries for BeyondMimic.

The original relative-body and global-anchor rewards/observations remain
unchanged. Existing soft anchor rewards do not certify paired combat geometry.
"""
import numpy as np
import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.math import quat_apply,quat_inv,quat_mul,yaw_quat
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


class MultiReferenceMotionCommand(MotionCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        with np.load(cfg.motion_file,allow_pickle=False) as z:
            starts=z['clip_starts'];ends=z['clip_ends']
            if len(starts)==0 or starts[0]!=0 or ends[-1]!=self.motion.time_step_total:
                raise ValueError('Invalid bundle bounds')
            if not np.array_equal(starts[1:],ends[:-1]) or np.any(ends-starts<4):
                raise ValueError('Invalid clip partition')
            self.clip_ids=list(z['clip_ids'])
        self.clip_starts=torch.tensor(starts,device=self.device,dtype=torch.long)
        self.clip_ends=torch.tensor(ends,device=self.device,dtype=torch.long)
        self.selected_clip=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.sample_counts=torch.zeros(len(starts),device=self.device,dtype=torch.long)

    def _adaptive_sampling(self,env_ids):
        clips=torch.randint(len(self.clip_starts),(len(env_ids),),device=self.device)
        self.selected_clip[env_ids]=clips
        # Leave two usable frames before the explicit end-of-clip timeout.
        length=self.clip_ends[clips]-self.clip_starts[clips]-2
        offset=(torch.rand(len(env_ids),device=self.device)*length).long()
        self.time_steps[env_ids]=self.clip_starts[clips]+offset
        self.sample_counts+=torch.bincount(clips,minlength=len(self.clip_starts))

    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        if len(env_ids):self._refresh_alignment()

    def _refresh_alignment(self):
        # Match upstream alignment exactly, without advancing reference time.
        anchor=self.anchor_pos_w[:,None,:]
        position=self.robot_anchor_pos_w[:,None,:].clone()
        position[...,2]=anchor[...,2]
        rotation=yaw_quat(quat_mul(self.robot_anchor_quat_w,quat_inv(self.anchor_quat_w)))[:,None,:].repeat(1,len(self.cfg.body_names),1)
        self.body_pos_relative_w=position+quat_apply(rotation,self.body_pos_w-anchor)
        self.body_quat_relative_w=quat_mul(rotation,self.body_quat_w)

    def _update_command(self):
        # Defensive cap: even an external caller cannot enter another clip.
        end=self.clip_ends[self.selected_clip]
        self.time_steps=torch.minimum(self.time_steps+1,end-1)-1
        super()._update_command()
        if torch.any(self.time_steps<self.clip_starts[self.selected_clip]) or torch.any(self.time_steps>=end):
            raise RuntimeError('Reference crossed a clip boundary')


def clip_finished(env,command_name='motion'):
    command=env.command_manager.get_term(command_name)
    return command.time_steps>=command.clip_ends[command.selected_clip]-2


def configure_multireference(cfg):
    cfg.commands.motion.class_type=MultiReferenceMotionCommand
    cfg.terminations.reference_clip_end=TerminationTermCfg(func=clip_finished,time_out=True)
