from types import SimpleNamespace

import torch

from protomotions.envs.motion_manager.overlapping_clip_motion_manager import (
    OverlappingClipMotionManager,
)


def _manager(full_probability: float):
    manager = OverlappingClipMotionManager.__new__(OverlappingClipMotionManager)
    manager.clip_attempts = torch.tensor([20, 8, 4], dtype=torch.long)
    manager.clip_failures = torch.tensor([2, 5, 4], dtype=torch.long)
    manager.full_motion_clip_id = 2 if full_probability > 0.0 else None
    manager.config = SimpleNamespace(
        max_failure_weight_ratio=4.0,
        failure_sampling_mix=0.8,
        full_motion_sampling_probability=full_probability,
    )
    return manager


def test_full_motion_probability_has_explicit_probability_mass():
    probabilities = _manager(0.25)._sampling_probabilities()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(probabilities[2], torch.tensor(0.25))
    torch.testing.assert_close(probabilities[:2].sum(), torch.tensor(0.75))


def test_zero_full_motion_probability_preserves_all_window_sampling():
    probabilities = _manager(0.0)._sampling_probabilities()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert torch.all(probabilities > 0.0)
