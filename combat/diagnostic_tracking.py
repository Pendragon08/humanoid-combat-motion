"""Clip-level failure sampling; frame-level sampling remains uniform."""
import torch
from combat.multireference_tracking import MultiReferenceMotionCommand
from combat.diagnostic_sampling import bounded_probabilities, sample_frames


class FailureRateMotionCommand(MultiReferenceMotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.failure_rates = torch.full((len(self.clip_ids),), .5, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _adaptive_sampling(self, env_ids):
        if len(env_ids) == 0:
            return
        old_ids = env_ids[self.started[env_ids]]
        old_clips = self.selected_clip[old_ids]
        counts = torch.bincount(old_clips, minlength=len(self.clip_ids))
        failed = self._env.termination_manager.terminated[old_ids].float()
        totals = torch.zeros_like(self.failure_rates).scatter_add_(0, old_clips, failed)
        valid = counts > 0
        self.failure_rates[valid] = .9*self.failure_rates[valid] + .1*totals[valid]/counts[valid]
        probabilities = bounded_probabilities(self.failure_rates)
        clips = torch.multinomial(probabilities, len(env_ids), replacement=True)
        self.selected_clip[env_ids] = clips
        self.time_steps[env_ids] = sample_frames(self.clip_starts, self.clip_ends, clips)
        self.sample_counts += torch.bincount(clips, minlength=len(self.clip_ids))
        self.started[env_ids] = True
