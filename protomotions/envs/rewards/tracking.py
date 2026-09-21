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
"""Tracking reward compute kernels for motion imitation.

Pure tensor functions (kernels) for computing tracking rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.tracking import compute_gt_rew
    
    reward_components = {
        "gt_rew": MdpComponent(
            compute_func=compute_gt_rew,
            dynamic_vars={
                "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
                "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            },
            static_params={"coefficient": -100.0},
        ),
    }

Includes:
- Standard AMP/DeepMimic-style tracking rewards (gt, gr, gv, gav, rh)
- BeyondMimic-style rewards (global/relative position, orientation, velocity)
"""

import torch
from torch import Tensor
from typing import List, Optional

from protomotions.utils.rotations import (
    quat_angle_diff_norm,
    calc_heading_quat_inv,
    quat_rotate,
    quat_mul,
)
from protomotions.envs.rewards.base import mean_squared_error_exp, rotation_error_exp


# =============================================================================
# Standard Tracking Reward Kernels
# =============================================================================

def compute_gt_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    coefficient: float = -100.0,
) -> Tensor:
    """Position tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        coefficient,
    )


def compute_reliability_blended_position_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    reference_reliability: Tensor,
    global_coefficient: float = -25.0,
    relative_coefficient: float = -25.0,
    minimum_absolute_weight: float = 0.05,
    gravity_axis_only: bool = False,
) -> Tensor:
    """Track absolute position only in proportion to physical confidence.

    The low-confidence fallback is root-relative body-position tracking, not a
    zero reward.  This preserves the observed articulation while allowing
    physics to reject a floating or penetrating global root trajectory.
    """
    if not 0.0 <= minimum_absolute_weight <= 1.0:
        raise ValueError("minimum_absolute_weight must lie in [0, 1]")
    reliability = reference_reliability.reshape(-1).clamp(0.0, 1.0)
    absolute_weight = minimum_absolute_weight + (
        1.0 - minimum_absolute_weight
    ) * reliability
    if gravity_axis_only:
        vertical_offset = (
            current_rigid_body_pos[:, :1, 2:3]
            - ref_rigid_body_pos[:, :1, 2:3]
        ) * (1.0 - absolute_weight[:, None, None])
        root_offset = torch.cat(
            (
                torch.zeros_like(vertical_offset).expand(-1, -1, 2),
                vertical_offset,
            ),
            dim=-1,
        )
        gated_reference = ref_rigid_body_pos + root_offset
        return mean_squared_error_exp(
            current_rigid_body_pos,
            gated_reference,
            global_coefficient,
        )
    absolute_reward = mean_squared_error_exp(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        global_coefficient,
    )
    current_relative = (
        current_rigid_body_pos - current_rigid_body_pos[:, :1, :]
    )
    reference_relative = ref_rigid_body_pos - ref_rigid_body_pos[:, :1, :]
    relative_reward = mean_squared_error_exp(
        current_relative,
        reference_relative,
        relative_coefficient,
    )
    return absolute_weight * absolute_reward + (
        1.0 - absolute_weight
    ) * relative_reward


def compute_terrain_feasible_gt_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
    coefficient: float = -25.0,
    penetration_margin: float = 0.02,
    penetration_scale: float = 0.05,
    min_z_weight: float = 0.05,
) -> Tensor:
    """Absolute body-position reward robust to an infeasible reference height.

    XY tracking is unchanged.  The Z error of a reference body is smoothly
    downweighted only when that body's origin lies below the local terrain plus
    ``penetration_margin``.  With no penetration this is exactly the standard
    global-position exponential MSE reward.

    This does not move or rewrite the reference motion.  It only prevents an
    impossible below-terrain Z target from dominating the policy gradient.
    """
    if penetration_scale <= 0:
        raise ValueError("penetration_scale must be positive.")
    if not 0.0 <= min_z_weight <= 1.0:
        raise ValueError("min_z_weight must be in [0, 1].")

    if ref_ground_heights.ndim == ref_rigid_body_pos.ndim:
        ref_ground_heights = ref_ground_heights.squeeze(-1)
    penetration = (
        ref_ground_heights + penetration_margin - ref_rigid_body_pos[..., 2]
    ).clamp_min(0.0)
    feasible_weight = torch.exp(-penetration / penetration_scale)
    z_weight = min_z_weight + (1.0 - min_z_weight) * feasible_weight

    squared_error = (current_rigid_body_pos - ref_rigid_body_pos).pow(2)
    weighted_error = torch.cat(
        [squared_error[..., :2], squared_error[..., 2:3] * z_weight.unsqueeze(-1)],
        dim=-1,
    )
    return torch.exp(coefficient * weighted_error.mean(dim=(-2, -1)))


def compute_blended_terrain_feasible_gt_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
    safe_reset_reward_lift: Tensor,
    coefficient: float = -25.0,
    penetration_margin: float = 0.02,
    penetration_scale: float = 0.05,
    min_z_weight: float = 0.05,
) -> Tensor:
    """Terrain-feasible tracking with a smooth safe-reset target transition."""
    blended_ref = torch.cat(
        (
            ref_rigid_body_pos[..., :2],
            ref_rigid_body_pos[..., 2:3]
            + safe_reset_reward_lift[:, None, None],
        ),
        dim=-1,
    )
    return compute_terrain_feasible_gt_rew(
        current_rigid_body_pos=current_rigid_body_pos,
        ref_rigid_body_pos=blended_ref,
        ref_ground_heights=ref_ground_heights,
        coefficient=coefficient,
        penetration_margin=penetration_margin,
        penetration_scale=penetration_scale,
        min_z_weight=min_z_weight,
    )


def compute_blended_rh_rew(
    current_root_height: Tensor,
    ref_rigid_body_pos: Tensor,
    safe_reset_reward_lift: Tensor,
    coefficient: float = -20.0,
) -> Tensor:
    """Root-height target consistent with the smooth safe-reset transition."""
    ref_root_height = (
        ref_rigid_body_pos[:, 0, 2] + safe_reset_reward_lift
    )
    return mean_squared_error_exp(
        current_root_height,
        ref_root_height,
        coefficient,
    )


def _compute_reference_feasible_lift(
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
    clearance_margin: float,
    max_lift: float,
) -> Tensor:
    """Return the minimum uniform Z lift that clears all reference body origins."""
    if ref_ground_heights.ndim == ref_rigid_body_pos.ndim:
        ref_ground_heights = ref_ground_heights.squeeze(-1)
    required_lift = (
        ref_ground_heights + clearance_margin - ref_rigid_body_pos[..., 2]
    ).amax(dim=-1).clamp_min(0.0)
    if max_lift > 0.0:
        required_lift = required_lift.clamp_max(max_lift)
    return required_lift


def compute_terrain_projected_gt_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
    coefficient: float = -25.0,
    clearance_margin: float = 0.02,
    max_lift: float = 0.15,
) -> Tensor:
    """Track the closest uniformly lifted, terrain-feasible reference pose.

    Safe reference reset lifts the complete simulated character.  Comparing that
    state against the original (possibly penetrating) global reference makes the
    reward immediately command the character back into the terrain.  This kernel
    applies the same minimum uniform lift to the *reward target only*.  MotionLib,
    future-pose observations, exported reference data, and visualization markers
    remain unchanged.
    """
    lift = _compute_reference_feasible_lift(
        ref_rigid_body_pos,
        ref_ground_heights,
        clearance_margin,
        max_lift,
    )
    projected_ref = torch.cat(
        (
            ref_rigid_body_pos[..., :2],
            ref_rigid_body_pos[..., 2:3] + lift[:, None, None],
        ),
        dim=-1,
    )
    return mean_squared_error_exp(
        current_rigid_body_pos,
        projected_ref,
        coefficient,
    )


def compute_terrain_projected_rh_rew(
    current_root_height: Tensor,
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
    coefficient: float = -20.0,
    clearance_margin: float = 0.02,
    max_lift: float = 0.15,
) -> Tensor:
    """Root-height reward consistent with ``compute_terrain_projected_gt_rew``."""
    lift = _compute_reference_feasible_lift(
        ref_rigid_body_pos,
        ref_ground_heights,
        clearance_margin,
        max_lift,
    )
    projected_root_height = ref_rigid_body_pos[:, 0, 2] + lift
    return mean_squared_error_exp(
        current_root_height,
        projected_root_height,
        coefficient,
    )


def compute_reference_feasible_lift(
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
    clearance_margin: float = 0.02,
    max_lift: float = 0.15,
) -> Tensor:
    """Expose the per-frame feasible-reference lift as a zero-weight metric."""
    return _compute_reference_feasible_lift(
        ref_rigid_body_pos,
        ref_ground_heights,
        clearance_margin,
        max_lift,
    )


def compute_reference_penetration_depth(
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
) -> Tensor:
    """Mean positive reference-body penetration depth, in meters."""
    if ref_ground_heights.ndim == ref_rigid_body_pos.ndim:
        ref_ground_heights = ref_ground_heights.squeeze(-1)
    return (ref_ground_heights - ref_rigid_body_pos[..., 2]).clamp_min(0.0).mean(-1)


def compute_reference_penetration_fraction(
    ref_rigid_body_pos: Tensor,
    ref_ground_heights: Tensor,
) -> Tensor:
    """Fraction of reference body origins below their local terrain height."""
    if ref_ground_heights.ndim == ref_rigid_body_pos.ndim:
        ref_ground_heights = ref_ground_heights.squeeze(-1)
    return (ref_rigid_body_pos[..., 2] < ref_ground_heights).float().mean(-1)


def compute_gr_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    coefficient: float = -5.0,
) -> Tensor:
    """Rotation tracking reward (exponential quaternion error).
    
    Args:
        current_rigid_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return rotation_error_exp(
        current_rigid_body_rot,
        ref_rigid_body_rot,
        coefficient,
    )


def _reference_angular_reliability(
    ref_rigid_body_ang_vel: Tensor,
    angular_speed_soft: float,
    angular_speed_scale: float,
    min_reliability: float,
) -> Tensor:
    if angular_speed_scale <= 0.0:
        raise ValueError("angular_speed_scale must be positive.")
    if not 0.0 <= min_reliability <= 1.0:
        raise ValueError("min_reliability must be in [0, 1].")
    speed = torch.linalg.vector_norm(ref_rigid_body_ang_vel, dim=-1)
    excess = (speed - angular_speed_soft).clamp_min(0.0)
    confidence = torch.exp(-torch.square(excess / angular_speed_scale))
    return min_reliability + (1.0 - min_reliability) * confidence


def compute_reliability_weighted_gr_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    coefficient: float = -5.0,
    angular_speed_soft: float = 4.0,
    angular_speed_scale: float = 2.0,
    min_reliability: float = 0.1,
) -> Tensor:
    """Robust global orientation reward for noisy video references.

    Below ``angular_speed_soft`` this is exactly the standard reward. Bodies
    whose reference angular speed is implausibly high retain only a smooth,
    bounded fraction of their tracking weight instead of forcing a twist.
    """
    reliability = _reference_angular_reliability(
        ref_rigid_body_ang_vel,
        angular_speed_soft,
        angular_speed_scale,
        min_reliability,
    )
    error = quat_angle_diff_norm(
        current_rigid_body_rot, ref_rigid_body_rot, w_last=True
    )
    weighted_error = (error * reliability).sum(-1) / reliability.sum(-1).clamp_min(
        1e-6
    )
    return torch.exp(coefficient * weighted_error)


def compute_gv_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    coefficient: float = -0.5,
) -> Tensor:
    """Velocity tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_vel: Current body velocities [num_envs, num_bodies, 3].
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_vel,
        ref_rigid_body_vel,
        coefficient,
    )


def compute_reliability_blended_velocity_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    reference_reliability: Tensor,
    global_coefficient: float = -0.5,
    relative_coefficient: float = -0.5,
    minimum_absolute_weight: float = 0.05,
    gravity_axis_only: bool = False,
) -> Tensor:
    """Track root translation velocity only when its observation is trusted."""
    if not 0.0 <= minimum_absolute_weight <= 1.0:
        raise ValueError("minimum_absolute_weight must lie in [0, 1]")
    reliability = reference_reliability.reshape(-1).clamp(0.0, 1.0)
    absolute_weight = minimum_absolute_weight + (
        1.0 - minimum_absolute_weight
    ) * reliability
    if gravity_axis_only:
        vertical_offset = (
            current_rigid_body_vel[:, :1, 2:3]
            - ref_rigid_body_vel[:, :1, 2:3]
        ) * (1.0 - absolute_weight[:, None, None])
        root_offset = torch.cat(
            (
                torch.zeros_like(vertical_offset).expand(-1, -1, 2),
                vertical_offset,
            ),
            dim=-1,
        )
        gated_reference = ref_rigid_body_vel + root_offset
        return mean_squared_error_exp(
            current_rigid_body_vel,
            gated_reference,
            global_coefficient,
        )
    absolute_reward = mean_squared_error_exp(
        current_rigid_body_vel,
        ref_rigid_body_vel,
        global_coefficient,
    )
    current_relative = current_rigid_body_vel - current_rigid_body_vel[:, :1, :]
    reference_relative = ref_rigid_body_vel - ref_rigid_body_vel[:, :1, :]
    relative_reward = mean_squared_error_exp(
        current_relative,
        reference_relative,
        relative_coefficient,
    )
    return absolute_weight * absolute_reward + (
        1.0 - absolute_weight
    ) * relative_reward


def compute_contact_transition_vertical_velocity_error(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    future_ref_root_vel: Tensor,
    current_ref_contacts: Tensor,
    future_reference_reliability: Tensor,
    support_body_ids: Optional[List[int]] = None,
    vertical_speed_scale: float = 3.0,
    horizon_weights: Optional[List[float]] = None,
    huber_delta: float = 0.5,
    max_error: float = 5.0,
) -> Tensor:
    """Dense, non-saturating take-off velocity tracking error.

    Generic exponential velocity rewards become nearly flat after a large
    noisy-reference miss.  This term is active only while a reference support
    contact precedes a reliable upward transition, and applies Smooth-L1 to
    the *current* root vertical velocity.  It therefore supplies timing-aware
    credit for generating support impulse without prescribing torques, adding
    external forces, or rewarding an early jump toward a distant peak.
    """
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    if max_error <= 0.0:
        raise ValueError("max_error must be positive")
    from protomotions.envs.obs.target_poses import (
        build_contact_conditioned_takeoff_demand,
    )

    demand = build_contact_conditioned_takeoff_demand(
        ref_rigid_body_vel,
        future_ref_root_vel,
        current_ref_contacts,
        future_reference_reliability,
        support_body_ids,
        vertical_speed_scale,
        horizon_weights,
    )[:, 0]
    velocity_error = (
        current_rigid_body_vel[:, 0, 2] - ref_rigid_body_vel[:, 0, 2]
    ).abs()
    smooth_l1 = torch.where(
        velocity_error < huber_delta,
        0.5 * velocity_error.square() / huber_delta,
        velocity_error - 0.5 * huber_delta,
    )
    return demand * smooth_l1.clamp(max=max_error)


def compute_gav_rew(
    current_rigid_body_ang_vel: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    coefficient: float = -0.1,
) -> Tensor:
    """Angular velocity tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_ang_vel: Current angular velocities [num_envs, num_bodies, 3].
        ref_rigid_body_ang_vel: Reference angular velocities [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_ang_vel,
        ref_rigid_body_ang_vel,
        coefficient,
    )


def compute_rh_rew(
    current_root_height: Tensor,
    ref_rigid_body_pos: Tensor,
    coefficient: float = -100.0,
) -> Tensor:
    """Root height tracking reward (exponential MSE).
    
    Args:
        current_root_height: Current root height [num_envs] or [num_envs, 1].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    # Extract reference root height (z-coordinate of root body)
    ref_root_height = ref_rigid_body_pos[:, 0, 2]
    
    return mean_squared_error_exp(
        current_root_height,
        ref_root_height,
        coefficient,
    )


# =============================================================================
# BeyondMimic-style Reward Kernels
# =============================================================================

def compute_global_position_error_exp(
    x: Tensor,
    ref_x: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Position error: exp(-||x - ref_x||^2 / sigma^2).
    
    Args:
        x: Current positions [num_envs, num_bodies, 3] or [num_envs, 3].
        ref_x: Reference positions (same shape as x).
        sigma: Gaussian kernel width.
        indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward tensor [num_envs].
    """
    if indices is not None and x.dim() == 3:
        x = x[:, indices]
        ref_x = ref_x[:, indices]

    error = (x - ref_x).pow(2).sum(dim=-1)
    if error.dim() == 2:
        error = error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def compute_global_anchor_pos_rew(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
    sigma: float = 0.3,
) -> Tensor:
    """Global anchor position reward (BeyondMimic style).
    
    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||anchor_pos - ref_anchor_pos||^2 / sigma^2).
    """
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    return compute_global_position_error_exp(current_anchor_pos, ref_anchor_pos, sigma)


def compute_global_orientation_error_exp(
    q: Tensor,
    ref_q: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Orientation error: exp(-angle_diff^2 / sigma^2).
    
    Args:
        q: Current orientations [num_envs, num_bodies, 4] or [num_envs, 4] (w-last).
        ref_q: Reference orientations (same shape as q).
        sigma: Gaussian kernel width.
        indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward tensor [num_envs].
    """
    if indices is not None and q.dim() == 3:
        q = q[:, indices]
        ref_q = ref_q[:, indices]

    error = quat_angle_diff_norm(q, ref_q, w_last=True)
    if error.dim() == 2:
        error = error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def compute_global_anchor_ori_rew(
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
    sigma: float = 0.4,
) -> Tensor:
    """Global anchor orientation reward (BeyondMimic style).
    
    Args:
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-angle_diff^2 / sigma^2).
    """
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    return compute_global_orientation_error_exp(current_anchor_rot, ref_anchor_rot, sigma)


def compute_relative_body_pos_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    current_anchor_pos: Tensor,
    anchor_idx: int,
    sigma: float = 0.3,
    body_indices: Optional[Tensor] = None,
) -> Tensor:
    """Relative body position reward (BeyondMimic style).
    
    Computes reward based on body positions relative to anchor in anchor's local frame.
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        current_anchor_pos: Current anchor position [num_envs, 3].
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
        body_indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward: exp(-||rel_pos - ref_rel_pos||^2 / sigma^2).
    """
    # Extract reference anchor pos and rot
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    # Compute heading rotations (yaw-only)
    current_heading_rot_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_rot_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute relative positions in world frame
    current_rel_pos = current_rigid_body_pos - current_anchor_pos.unsqueeze(1)
    ref_rel_pos = ref_rigid_body_pos - ref_anchor_pos.unsqueeze(1)
    
    # Rotate to anchor's local frame
    current_rel_pos_flat = current_rel_pos.reshape(-1, 3)
    current_heading_rot_inv_exp = current_heading_rot_inv.unsqueeze(1).expand(
        -1, current_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    current_rel_pos_local = quat_rotate(
        current_heading_rot_inv_exp, current_rel_pos_flat, w_last=True
    ).reshape(current_rigid_body_pos.shape)
    
    ref_rel_pos_flat = ref_rel_pos.reshape(-1, 3)
    ref_heading_rot_inv_exp = ref_heading_rot_inv.unsqueeze(1).expand(
        -1, ref_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    ref_rel_pos_local = quat_rotate(
        ref_heading_rot_inv_exp, ref_rel_pos_flat, w_last=True
    ).reshape(ref_rigid_body_pos.shape)
    
    return compute_global_position_error_exp(
        current_rel_pos_local, ref_rel_pos_local, sigma, body_indices
    )


def compute_relative_body_ori_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    current_anchor_rot: Tensor,
    anchor_idx: int,
    sigma: float = 0.4,
    body_indices: Optional[Tensor] = None,
) -> Tensor:
    """Relative body orientation reward (BeyondMimic style).
    
    Computes reward based on body orientations relative to anchor.
    
    Args:
        current_rigid_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
        body_indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward: exp(-angle_diff^2 / sigma^2).
    """
    # Extract reference anchor rotation
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    # Compute heading rotations (yaw-only)
    current_heading_rot_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_rot_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute relative rotations
    current_heading_rot_inv_exp = current_heading_rot_inv.unsqueeze(1).expand(
        -1, current_rigid_body_rot.shape[1], -1
    )
    current_rel_rot = quat_mul(current_heading_rot_inv_exp, current_rigid_body_rot, w_last=True)
    
    ref_heading_rot_inv_exp = ref_heading_rot_inv.unsqueeze(1).expand(
        -1, ref_rigid_body_rot.shape[1], -1
    )
    ref_rel_rot = quat_mul(ref_heading_rot_inv_exp, ref_rigid_body_rot, w_last=True)
    
    return compute_global_orientation_error_exp(
        current_rel_rot, ref_rel_rot, sigma, body_indices
    )


def compute_reliability_weighted_relative_body_ori_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    current_anchor_rot: Tensor,
    anchor_idx: int,
    sigma: float = 0.4,
    angular_speed_soft: float = 4.0,
    angular_speed_scale: float = 2.0,
    min_reliability: float = 0.1,
) -> Tensor:
    """Heading-relative orientation reward with body-wise reliability."""
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    current_heading_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    current_rel_rot = quat_mul(
        current_heading_inv[:, None, :].expand_as(current_rigid_body_rot),
        current_rigid_body_rot,
        w_last=True,
    )
    ref_rel_rot = quat_mul(
        ref_heading_inv[:, None, :].expand_as(ref_rigid_body_rot),
        ref_rigid_body_rot,
        w_last=True,
    )
    reliability = _reference_angular_reliability(
        ref_rigid_body_ang_vel,
        angular_speed_soft,
        angular_speed_scale,
        min_reliability,
    )
    error = quat_angle_diff_norm(current_rel_rot, ref_rel_rot, w_last=True)
    weighted_error = (error * reliability).sum(-1) / reliability.sum(-1).clamp_min(
        1e-6
    )
    return torch.exp(-weighted_error / (sigma**2))


def compute_global_body_lin_vel_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    sigma: float = 1.0,
) -> Tensor:
    """Global body linear velocity reward (BeyondMimic style).
    
    Args:
        current_rigid_body_vel: Current body velocities [num_envs, num_bodies, 3].
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||vel - ref_vel||^2 / sigma^2).
    """
    return compute_global_position_error_exp(current_rigid_body_vel, ref_rigid_body_vel, sigma)


def compute_global_body_ang_vel_rew(
    current_rigid_body_ang_vel: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    sigma: float = 3.14,
) -> Tensor:
    """Global body angular velocity reward (BeyondMimic style).
    
    Args:
        current_rigid_body_ang_vel: Current angular velocities [num_envs, num_bodies, 3].
        ref_rigid_body_ang_vel: Reference angular velocities [num_envs, num_bodies, 3].
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||ang_vel - ref_ang_vel||^2 / sigma^2).
    """
    return compute_global_position_error_exp(
        current_rigid_body_ang_vel, ref_rigid_body_ang_vel, sigma
    )


__all__ = [
    # Standard tracking rewards
    "compute_gt_rew",
    "compute_reliability_blended_position_rew",
    "compute_terrain_feasible_gt_rew",
    "compute_reference_penetration_depth",
    "compute_reference_penetration_fraction",
    "compute_gr_rew",
    "compute_gv_rew",
    "compute_reliability_blended_velocity_rew",
    "compute_gav_rew",
    "compute_rh_rew",
    # BeyondMimic-style rewards
    "compute_global_position_error_exp",
    "compute_global_anchor_pos_rew",
    "compute_global_orientation_error_exp",
    "compute_global_anchor_ori_rew",
    "compute_relative_body_pos_rew",
    "compute_relative_body_ori_rew",
    "compute_global_body_lin_vel_rew",
    "compute_global_body_ang_vel_rew",
]
