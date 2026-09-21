# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Regularization reward compute kernels.

Pure tensor functions (kernels) for computing regularization rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.regularization import compute_action_smoothness
    
    reward_components = {
        "action_smoothness": MdpComponent(
            compute_func=compute_action_smoothness,
            dynamic_vars={
                "current_processed_action": EnvContext.current_processed_action,
                "previous_processed_action": EnvContext.previous_processed_action,
            },
        ),
    }

Includes:
- Action smoothness (L2 and Log-Mean-Exp variants)
- Power consumption
- Joint limit violations
- Contact matching
- Contact force change penalties
"""

import torch
from torch import Tensor
from typing import List, Optional

from protomotions.envs.rewards.base import power_consumption_sum, delta_norm, delta_logmeanexp


# =============================================================================
# Regularization Reward Kernels
# =============================================================================

def compute_action_smoothness(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
) -> Tensor:
    """Action smoothness reward (L2 norm of processed action changes).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_norm(current_processed_action, previous_processed_action)


def compute_reset_aware_action_smoothness(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    episode_progress: Tensor,
    ignore_first_steps: int = 1,
) -> Tensor:
    """Action delta that ignores only invalid zero history after a reset.

    ``previous_processed_action`` is artificially zero before the first control
    step.  From the second step onward it is a real policy action and should be
    regularized, including during the remaining reset grace period.
    """
    penalty = delta_norm(current_processed_action, previous_processed_action)
    if ignore_first_steps <= 0:
        return penalty
    return torch.where(episode_progress <= ignore_first_steps, 0.0, penalty)


def compute_reset_aware_action_acceleration(
    current_processed_action: Tensor,
    historical_processed_actions: Tensor,
    episode_progress: Tensor,
    indices: Optional[List[int]] = None,
    ignore_first_steps: int = 2,
) -> Tensor:
    """Penalize the second finite difference of processed policy actions.

    Unlike first-difference smoothing, this permits a sustained correction but
    penalizes alternating commands, which are a common source of visible joint
    jerk. ``historical_processed_actions[:, 0]`` is the previous action and
    index 1 is the action before that.  The first two post-reset steps are
    ignored because their history is synthetic.
    """
    if historical_processed_actions.shape[1] < 2:
        raise ValueError(
            "Action acceleration requires env.num_state_history_steps >= 2."
        )
    previous = historical_processed_actions[:, 0]
    previous_previous = historical_processed_actions[:, 1]
    acceleration = current_processed_action - 2.0 * previous + previous_previous
    if indices is not None:
        acceleration = acceleration[:, indices]
    penalty = torch.linalg.vector_norm(acceleration, dim=-1)
    if ignore_first_steps <= 0:
        return penalty
    return torch.where(episode_progress <= ignore_first_steps, 0.0, penalty)


def relax_regularization_with_demand(
    penalty: Tensor,
    demand: Tensor,
    minimum_multiplier: float = 0.1,
) -> Tensor:
    """Relax a smoothness penalty only where dynamics require an impulse."""
    if not 0.0 <= minimum_multiplier <= 1.0:
        raise ValueError("minimum_multiplier must lie in [0, 1]")
    demand = demand.reshape(demand.shape[0], -1)
    if demand.shape[1] != 1 or demand.shape[0] != penalty.shape[0]:
        raise ValueError("demand must be scalar per environment")
    multiplier = 1.0 - (1.0 - minimum_multiplier) * demand[:, 0].clamp(0.0, 1.0)
    return penalty * multiplier


def compute_demand_relaxed_action_smoothness(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    episode_progress: Tensor,
    current_ref_body_vel: Tensor,
    future_ref_root_vel: Tensor,
    current_ref_contacts: Tensor,
    future_reference_reliability: Tensor,
    support_body_ids: Optional[List[int]] = None,
    vertical_speed_scale: float = 3.0,
    horizon_weights: Optional[List[float]] = None,
    minimum_multiplier: float = 0.1,
    ignore_first_steps: int = 1,
) -> Tensor:
    """Reset-aware action delta with contact-transition-aware relaxation."""
    from protomotions.envs.obs.target_poses import (
        build_contact_conditioned_takeoff_demand,
    )

    penalty = compute_reset_aware_action_smoothness(
        current_processed_action,
        previous_processed_action,
        episode_progress,
        ignore_first_steps,
    )
    demand = build_contact_conditioned_takeoff_demand(
        current_ref_body_vel,
        future_ref_root_vel,
        current_ref_contacts,
        future_reference_reliability,
        support_body_ids,
        vertical_speed_scale,
        horizon_weights,
    )
    return relax_regularization_with_demand(
        penalty, demand, minimum_multiplier=minimum_multiplier
    )


def compute_demand_relaxed_action_acceleration(
    current_processed_action: Tensor,
    historical_processed_actions: Tensor,
    episode_progress: Tensor,
    current_ref_body_vel: Tensor,
    future_ref_root_vel: Tensor,
    current_ref_contacts: Tensor,
    future_reference_reliability: Tensor,
    indices: Optional[List[int]] = None,
    support_body_ids: Optional[List[int]] = None,
    vertical_speed_scale: float = 3.0,
    horizon_weights: Optional[List[float]] = None,
    minimum_multiplier: float = 0.1,
    ignore_first_steps: int = 2,
) -> Tensor:
    """Action second difference relaxed during required support impulses."""
    from protomotions.envs.obs.target_poses import (
        build_contact_conditioned_takeoff_demand,
    )

    penalty = compute_reset_aware_action_acceleration(
        current_processed_action,
        historical_processed_actions,
        episode_progress,
        indices,
        ignore_first_steps,
    )
    demand = build_contact_conditioned_takeoff_demand(
        current_ref_body_vel,
        future_ref_root_vel,
        current_ref_contacts,
        future_reference_reliability,
        support_body_ids,
        vertical_speed_scale,
        horizon_weights,
    )
    return relax_regularization_with_demand(
        penalty, demand, minimum_multiplier=minimum_multiplier
    )


def compute_reset_aware_relative_body_angular_jerk(
    current_rigid_body_ang_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor,
    episode_progress: Tensor,
    body_indices: List[int],
    parent_indices: List[int],
    ignore_first_steps: int = 2,
) -> Tensor:
    """Penalize step-wise angular jerk of selected joints.

    A joint's angular velocity is approximated by child minus parent rigid-body
    angular velocity. This removes global/root rotation before taking the
    second finite difference and directly targets oscillatory joint motion.
    The result is left in per-control-step units so reward magnitudes do not
    depend on a large ``dt**-2`` scale factor.
    """
    if len(body_indices) != len(parent_indices):
        raise ValueError("body_indices and parent_indices must have equal length.")
    if historical_rigid_body_ang_vel.shape[1] < 2:
        raise ValueError(
            "Relative body angular jerk requires env.num_state_history_steps >= 2."
        )

    current_relative = (
        current_rigid_body_ang_vel[:, body_indices]
        - current_rigid_body_ang_vel[:, parent_indices]
    )
    previous_all = historical_rigid_body_ang_vel[:, 0]
    previous_previous_all = historical_rigid_body_ang_vel[:, 1]
    previous_relative = previous_all[:, body_indices] - previous_all[:, parent_indices]
    previous_previous_relative = (
        previous_previous_all[:, body_indices]
        - previous_previous_all[:, parent_indices]
    )
    angular_jerk = (
        current_relative - 2.0 * previous_relative + previous_previous_relative
    )
    penalty = torch.linalg.vector_norm(angular_jerk, dim=-1).mean(dim=-1)
    if ignore_first_steps <= 0:
        return penalty
    return torch.where(episode_progress <= ignore_first_steps, 0.0, penalty)


def compute_action_smoothness_logmeanexp(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    beta: float = 3.0,
) -> Tensor:
    """Action smoothness using Log-Mean-Exp (soft L_infinity).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
        beta: Temperature parameter. Lower = more like mean, higher = more like max.
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_logmeanexp(current_processed_action, previous_processed_action, beta)


def compute_pow_rew(
    dof_forces: Tensor,
    dof_vel: Tensor,
    use_torque_squared: bool = False,
) -> Tensor:
    """Power consumption reward.
    
    Args:
        dof_forces: Joint forces/torques [num_envs, num_dofs].
        dof_vel: Joint velocities [num_envs, num_dofs].
        use_torque_squared: Whether to use torque squared instead of absolute.
    
    Returns:
        Power consumption tensor [num_envs].
    """
    return power_consumption_sum(dof_forces, dof_vel, use_torque_squared)


def compute_soft_pos_limit_rew(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
) -> Tensor:
    """Soft joint position limit penalty.
    
    Penalizes when joints approach or exceed limits.
    
    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower joint limits [num_dofs].
        dof_limits_upper: Upper joint limits [num_dofs].
    
    Returns:
        Penalty tensor [num_envs].
    """
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def compute_contact_match_rew(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    contact_body_ids: Tensor,
) -> Tensor:
    """Contact matching reward using foot contact bodies.
    
    Penalizes mismatch between simulated and reference foot contacts.
    Uses contact_body_ids (typically foot bodies).
    
    Args:
        sim_contacts: Simulated contact flags [num_envs, num_bodies].
        ref_contacts: Reference contact flags [num_envs, num_bodies].
        contact_body_ids: Indices of bodies to track contacts for [num_contact_bodies].
    
    Returns:
        Contact mismatch penalty tensor [num_envs].
    """
    sim_contacts_subset = sim_contacts[:, contact_body_ids]
    ref_contacts_subset = ref_contacts[:, contact_body_ids]
    return torch.abs(sim_contacts_subset.float() - ref_contacts_subset.float()).sum(dim=1)


def compute_contact_force_change_rew(
    current_contact_force_magnitudes: Tensor,
    prev_contact_force_magnitudes: Tensor,
    force_change_threshold: float = 30.0,
) -> Tensor:
    """Contact force change penalty.
    
    Penalizes sudden contact force changes above a threshold (impact penalty).
    
    Args:
        current_contact_force_magnitudes: Current contact forces [num_envs, num_bodies].
        prev_contact_force_magnitudes: Previous contact forces [num_envs, num_bodies].
        force_change_threshold: Force change below which impact is ignored.
    
    Returns:
        Total force change above threshold [num_envs].
    """
    force_changes = torch.abs(current_contact_force_magnitudes - prev_contact_force_magnitudes)
    force_changes = torch.clamp(force_changes - force_change_threshold, min=0)
    return force_changes.sum(dim=-1)


def compute_foot_sliding_rew(
    rigid_body_vel: Tensor,
    rigid_body_contacts: Tensor,
    contact_body_ids: Tensor,
) -> Tensor:
    """Horizontal foot speed while the simulator reports physical contact."""
    foot_vel_xy = rigid_body_vel[:, contact_body_ids, :2]
    foot_contacts = rigid_body_contacts[:, contact_body_ids].float()
    return (torch.linalg.vector_norm(foot_vel_xy, dim=-1) * foot_contacts).sum(-1)


# =============================================================================
# Helper Functions (used by kernels or for advanced use cases)
# =============================================================================

def joint_limit_violation(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of joint position limit violations.

    Penalizes positions outside [lower, upper] limits.

    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower limits [num_dofs].
        dof_limits_upper: Upper limits [num_dofs].
        indices: Optional DOF indices to subset.

    Returns:
        Total violation [num_envs].
    """
    if indices is not None:
        dof_pos = dof_pos[:, indices]
        dof_limits_lower = dof_limits_lower[indices]
        dof_limits_upper = dof_limits_upper[indices]

    below_lower = -(dof_pos - dof_limits_lower).clip(max=0.0)
    above_upper = (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(below_lower + above_upper, dim=1)


def contact_mismatch_sum(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of contact state mismatches.

    Computes sum(|sim_contacts - ref_contacts|).

    Args:
        sim_contacts: Simulated contacts [num_envs, num_bodies].
        ref_contacts: Reference contacts [num_envs, num_bodies].
        indices: Optional body indices to subset.

    Returns:
        Total mismatch [num_envs].
    """
    if indices is not None:
        sim_contacts = sim_contacts[:, indices]
        ref_contacts = ref_contacts[:, indices]

    return torch.abs(sim_contacts.float() - ref_contacts.float()).sum(dim=1)


def impact_force_penalty(
    current_forces: Tensor,
    previous_forces: Tensor,
    indices: Optional[Tensor] = None,
    threshold: float = 30.0,
) -> Tensor:
    """Sum of sudden contact force changes above a threshold (impact penalty).

    Penalizes abrupt force changes (both increases and decreases) that exceed
    the threshold. Small force changes below the threshold are ignored.

    Args:
        current_forces: Current contact forces [num_envs, num_bodies].
        previous_forces: Previous contact forces [num_envs, num_bodies].
        indices: Optional body indices to subset.
        threshold: Force change threshold below which changes are ignored (default: 30.0).

    Returns:
        Total force change above threshold [num_envs].
    """
    if indices is not None:
        current_forces = current_forces[:, indices]
        previous_forces = previous_forces[:, indices]

    force_changes = torch.abs(current_forces - previous_forces)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


__all__ = [
    # Main reward kernels
    "compute_action_smoothness",
    "compute_reset_aware_action_smoothness",
    "compute_reset_aware_action_acceleration",
    "compute_reset_aware_relative_body_angular_jerk",
    "compute_action_smoothness_logmeanexp",
    "compute_pow_rew",
    "compute_soft_pos_limit_rew",
    "compute_contact_match_rew",
    "compute_contact_force_change_rew",
    "compute_foot_sliding_rew",
    # Helper functions
    "joint_limit_violation",
    "contact_mismatch_sum",
    "impact_force_penalty",
]
