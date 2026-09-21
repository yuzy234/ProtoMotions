# SPDX-License-Identifier: Apache-2.0

import torch

from protomotions.envs.control.mimic_control import (
    select_reliability_adaptive_phase_rate,
)


def _select(current, candidates, reliability, previous=None):
    envs = current.shape[0]
    if previous is None:
        previous = torch.ones(envs)
    return select_reliability_adaptive_phase_rate(
        current,
        candidates,
        torch.tensor([1.0, 2.0, 3.0]),
        torch.as_tensor(reliability),
        previous,
        unreliable_below=0.25,
        reliable_above=0.75,
        contact_lock_reliability=0.8,
        pose_scale=0.2,
        prior_weight=0.25,
        change_weight=0.05,
    )


def test_low_reliability_can_advance_to_state_matching_candidate():
    current = torch.zeros(1, 2, 3)
    candidates = torch.ones(1, 3, 2, 3)
    candidates[:, 2] = current
    selected, preferred, pose_cost = _select(current, candidates, [0.0])
    assert selected.item() == 3.0
    assert preferred.item() == 3.0
    assert pose_cost.item() == 0.0


def test_reliable_contact_is_hard_locked_to_recorded_rate():
    current = torch.zeros(1, 2, 3)
    candidates = torch.ones(1, 3, 2, 3)
    candidates[:, 2] = current
    selected, preferred, _ = _select(current, candidates, [0.95])
    assert selected.item() == 1.0
    assert preferred.item() == 1.0


def test_phase_never_rewinds():
    current = torch.zeros(1, 2, 3)
    candidates = torch.zeros(1, 3, 2, 3)
    selected, _, _ = _select(current, candidates, [0.5])
    assert selected.item() >= 1.0
