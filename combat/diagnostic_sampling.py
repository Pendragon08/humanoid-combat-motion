"""Bounded failure-rate sampling for diagnostic multi-clip experiments."""
import torch


def bounded_probabilities(failure_rates):
    if failure_rates.ndim != 1 or failure_rates.numel() == 0:
        raise ValueError('Expected nonempty vector')
    if not torch.isfinite(failure_rates).all():
        raise ValueError('Nonfinite failure rates')
    weights = .5 + failure_rates.clamp(0, 1)
    return .2 / len(weights) + .8 * weights / weights.sum()


def sample_frames(starts, ends, clips):
    length = ends[clips] - starts[clips] - 2
    if torch.any(length <= 0):
        raise ValueError('Invalid clip lengths')
    return starts[clips] + (torch.rand(len(clips), device=starts.device) * length).long()
