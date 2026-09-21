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
"""Target pose building utilities for mimic environments.

Provides functions for building target pose observations from reference motions,
used for motion tracking and imitation learning.
"""

from typing import List, Optional, Union

import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.envs.obs.utils import select_step_indices


def build_contact_conditioned_takeoff_demand(
    current_ref_body_vel: Tensor,
    future_ref_root_vel: Tensor,
    current_ref_contacts: Tensor,
    future_reference_reliability: Tensor,
    support_body_ids: Optional[List[int]] = None,
    vertical_speed_scale: float = 3.0,
    horizon_weights: Optional[List[float]] = None,
) -> Tensor:
    """Measure a reliable, near-future launch demand while support still exists.

    The scalar is intended to condition *training exploration*, not the
    deterministic action mean.  A demand is emitted only when a designated
    reference support body is currently in contact and a reliable near-future
    root command requires both upward motion and an increase in vertical
    velocity.  Optional horizon weights prevent a distant jump in a long
    command window from injecting noise throughout an otherwise easy prefix.

    Returns:
        Tensor of shape ``[num_envs, 1]`` in ``[0, 1]``.
    """
    if current_ref_body_vel.ndim != 3 or current_ref_body_vel.shape[-1] != 3:
        raise ValueError("current_ref_body_vel must have shape [envs, bodies, 3]")
    if future_ref_root_vel.ndim != 3 or future_ref_root_vel.shape[-1] != 3:
        raise ValueError(
            "future_ref_root_vel must have shape [envs, future_steps, 3]"
        )
    if current_ref_contacts.ndim != 2:
        raise ValueError("current_ref_contacts must have shape [envs, bodies]")
    if future_reference_reliability.ndim != 2:
        raise ValueError(
            "future_reference_reliability must have shape [envs, future_steps]"
        )
    if current_ref_body_vel.shape[0] != future_ref_root_vel.shape[0]:
        raise ValueError("current and future root velocity batches must match")
    if current_ref_contacts.shape[0] != future_ref_root_vel.shape[0]:
        raise ValueError("contact and velocity batches must match")
    if future_reference_reliability.shape != future_ref_root_vel.shape[:2]:
        raise ValueError(
            "future_reference_reliability must match the future velocity horizon"
        )
    if vertical_speed_scale <= 0.0:
        raise ValueError("vertical_speed_scale must be positive")

    if support_body_ids is None:
        support_body_ids = [3, 4, 7, 8]
    if not support_body_ids:
        raise ValueError("support_body_ids cannot be empty")
    if (
        min(support_body_ids) < 0
        or max(support_body_ids) >= current_ref_contacts.shape[1]
    ):
        raise ValueError("support_body_ids contains an out-of-range body index")

    support = current_ref_contacts[:, support_body_ids].to(
        dtype=future_ref_root_vel.dtype
    ).amax(dim=-1)
    current_vertical = current_ref_body_vel[:, :1, 2]
    future_vertical = future_ref_root_vel[..., 2]
    upward_speed = torch.relu(future_vertical)
    upward_speed_gain = torch.relu(future_vertical - current_vertical)
    # Both conditions are needed: recovering from downward velocity without a
    # future upward command is not a launch, and high constant upward velocity
    # no longer needs additional support-phase exploration.
    launch_speed = torch.minimum(upward_speed, upward_speed_gain)
    trusted_launch_speed = (
        launch_speed * future_reference_reliability.clamp(0.0, 1.0)
    )

    if horizon_weights is not None:
        if len(horizon_weights) != future_ref_root_vel.shape[1]:
            raise ValueError(
                "horizon_weights must contain one value per future velocity step"
            )
        weights = future_ref_root_vel.new_tensor(horizon_weights)
        if bool(torch.any(weights < 0.0)):
            raise ValueError("horizon_weights must be non-negative")
        trusted_launch_speed = trusted_launch_speed * weights.unsqueeze(0)

    demand = trusted_launch_speed.amax(dim=1) / float(vertical_speed_scale)
    return (support * demand.clamp(0.0, 1.0)).unsqueeze(-1)


def build_reference_reliability(
    mimic_ref_vel: Tensor,
    mimic_ref_ang_vel: Tensor,
    linear_speed_soft: float = 4.0,
    angular_speed_soft: float = 6.0,
    linear_change_scale: float = 2.0,
    angular_change_scale: float = 4.0,
    min_reliability: float = 0.1,
) -> Tensor:
    """Estimate a causal confidence gate from a short future command window.

    Video-derived references occasionally contain one-frame velocity spikes.
    Absolute speed alone is not enough to identify them because fast limbs can
    be valid, so the score combines soft speed limits with disagreement among
    the future velocity samples.  The output is deliberately a single scalar
    per environment: it gates only the new temporal side path and never erases
    the pretrained tracker's original one-step command.
    """
    linear_speed = torch.linalg.vector_norm(mimic_ref_vel, dim=-1)
    angular_speed = torch.linalg.vector_norm(mimic_ref_ang_vel, dim=-1)
    speed_penalty = (
        torch.relu(linear_speed - linear_speed_soft) / linear_speed_soft
        + torch.relu(angular_speed - angular_speed_soft) / angular_speed_soft
    )

    if mimic_ref_vel.shape[1] > 1:
        linear_change = torch.linalg.vector_norm(
            mimic_ref_vel[:, 1:] - mimic_ref_vel[:, :-1], dim=-1
        )
        angular_change = torch.linalg.vector_norm(
            mimic_ref_ang_vel[:, 1:] - mimic_ref_ang_vel[:, :-1], dim=-1
        )
        change_penalty = (
            linear_change / linear_change_scale
            + angular_change / angular_change_scale
        ).mean(dim=(1, 2))
    else:
        change_penalty = speed_penalty.new_zeros(speed_penalty.shape[0])

    penalty = speed_penalty.mean(dim=(1, 2)) + change_penalty
    reliability = torch.exp(-penalty).clamp(min=min_reliability, max=1.0)
    return reliability.unsqueeze(-1)


def build_max_coords_target_poses_future_rel(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    mimic_ref_pos: Tensor,
    mimic_ref_rot: Tensor,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
):
    """Build target pose observations with relative deltas between consecutive future frames.

    Computes future target poses where each frame is expressed relative to the previous frame,
    providing incremental motion information for tracking.

    Args:
        current_state_body_pos: Current body positions [envs, bodies, 3]
        current_state_body_rot: Current body rotations [envs, bodies, 4]
        mimic_ref_pos: Reference body positions [envs, future_steps, bodies, 3]
        mimic_ref_rot: Reference body rotations [envs, future_steps, bodies, 4]
        w_last: If True, quaternions are in XYZW format, else WXYZ
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.

    Returns:
        Target pose observations [envs, features] in root-relative coordinates
    """

    num_envs = current_state_body_pos.shape[0]
    num_bodies = mimic_ref_pos.shape[2]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_pos = select_step_indices(mimic_ref_pos, future_steps)
        mimic_ref_rot = select_step_indices(mimic_ref_rot, future_steps)

    future_steps = mimic_ref_pos.shape[1]

    # Flatten reference tensors: [envs, future_steps, bodies, dim] -> [envs*future_steps, bodies, dim]
    ref_state_body_pos = mimic_ref_pos.reshape(-1, num_bodies, 3)
    ref_state_body_rot = mimic_ref_rot.reshape(-1, num_bodies, 4)

    reference_pos = mimic_ref_pos.clone().roll(shifts=1, dims=1)
    reference_pos[:, 0] = current_state_body_pos
    flat_reference_pos = reference_pos.reshape(ref_state_body_pos.shape)

    reference_rot = mimic_ref_rot.clone().roll(shifts=1, dims=1)
    reference_rot[:, 0] = current_state_body_rot
    flat_reference_rot = reference_rot.reshape(ref_state_body_rot.shape)

    reference_root_pos = flat_reference_pos[:, 0, :]
    reference_root_rot = flat_reference_rot[:, 0, :]

    heading_inv_rot = rotations.calc_heading_quat_inv(reference_root_rot, w_last)

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    pos_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_reference_pos.shape[1], 1)
    )
    rot_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_reference_rot.shape[1], 1)
    )
    pos_flat_heading_inv_rot = pos_heading_inv_rot_expand.reshape(
        pos_heading_inv_rot_expand.shape[0] * pos_heading_inv_rot_expand.shape[1],
        pos_heading_inv_rot_expand.shape[2],
    )

    reference_root_pos_expand = reference_root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    target_rel_body_pos = ref_state_body_pos - flat_reference_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(
        target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
        target_rel_body_pos.shape[2],
    )
    flat_target_rel_body_pos = rotations.quat_rotate(
        pos_flat_heading_inv_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (ref_state_body_pos - reference_root_pos_expand).reshape(
        ref_state_body_pos.shape[0] * ref_state_body_pos.shape[1],
        ref_state_body_pos.shape[2],
    )
    flat_target_body_pos = rotations.quat_rotate(
        pos_flat_heading_inv_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_reference_rot, w_last), ref_state_body_rot, w_last
    )
    target_rel_body_rot_obs = (
        rotations.quat_to_tan_norm(target_rel_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, future_steps, -1, 6)
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        rot_heading_inv_rot_expand, ref_state_body_rot, w_last
    )
    target_body_rot_obs = (
        rotations.quat_to_tan_norm(target_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, future_steps, -1, 6)
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    target_rel_body_pos = flat_target_rel_body_pos.reshape(
        num_envs, future_steps, -1, 3
    ).reshape(target_rel_body_pos.shape[0], -1)
    target_body_pos = flat_target_body_pos.reshape(
        num_envs, future_steps, -1, 3
    ).reshape(ref_state_body_pos.shape[0], -1)

    obs = torch.cat(
        (
            target_rel_body_pos,
            target_body_pos,
            target_rel_body_rot_obs,
            target_body_rot_obs,
        ),
        dim=-1,
    ).view(num_envs, -1)

    return obs


def build_max_coords_target_poses(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    current_state_body_vel: Tensor,
    current_state_body_ang_vel: Tensor,
    mimic_ref_pos: Tensor,
    mimic_ref_rot: Tensor,
    mimic_ref_vel: Tensor,
    mimic_ref_ang_vel: Tensor,
    with_velocities: bool,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
    with_relative: bool = True,
):
    """Build target pose observations in root-relative coordinates.

    Computes future target poses represented as both absolute (from root) and optionally
    relative (from current pose) transformations, in the root's heading-aligned frame.

    Args:
        current_state_body_pos: Current body positions [envs, bodies, 3]
        current_state_body_rot: Current body rotations [envs, bodies, 4]
        current_state_body_vel: Current body velocities [envs, bodies, 3]
        current_state_body_ang_vel: Current body angular velocities [envs, bodies, 3]
        mimic_ref_pos: Reference body positions [envs, future_steps, bodies, 3]
        mimic_ref_rot: Reference body rotations [envs, future_steps, bodies, 4]
        mimic_ref_vel: Reference body velocities [envs, future_steps, bodies, 3]
        mimic_ref_ang_vel: Reference body angular velocities [envs, future_steps, bodies, 3]
        with_velocities: If True, include velocity information
        w_last: If True, quaternions are in XYZW format, else WXYZ
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        with_relative: If True, include relative pose observations (pos_rel, rot_rel)

    Returns:
        Target pose observations [envs, features] with absolute and optionally relative pose info
    """
    num_envs = current_state_body_pos.shape[0]
    num_bodies = mimic_ref_pos.shape[2]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_pos = select_step_indices(mimic_ref_pos, future_steps)
        mimic_ref_rot = select_step_indices(mimic_ref_rot, future_steps)
        mimic_ref_vel = select_step_indices(mimic_ref_vel, future_steps)
        mimic_ref_ang_vel = select_step_indices(mimic_ref_ang_vel, future_steps)

    future_steps = mimic_ref_pos.shape[1]

    # Flatten reference tensors: [envs, future_steps, bodies, dim] -> [envs*future_steps, bodies, dim]
    ref_state_body_pos = mimic_ref_pos.reshape(-1, num_bodies, 3)
    ref_state_body_rot = mimic_ref_rot.reshape(-1, num_bodies, 4)
    ref_state_body_vel = mimic_ref_vel.reshape(-1, num_bodies, 3)
    ref_state_body_ang_vel = mimic_ref_ang_vel.reshape(-1, num_bodies, 3)

    expanded_body_pos = current_state_body_pos.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_pos.shape[1:]
    )
    expanded_body_rot = current_state_body_rot.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_rot.shape[1:]
    )

    flat_current_state_body_pos = expanded_body_pos.reshape(ref_state_body_pos.shape)
    flat_current_state_body_rot = expanded_body_rot.reshape(ref_state_body_rot.shape)

    root_pos = flat_current_state_body_pos[:, 0, :]
    root_rot = flat_current_state_body_rot[:, 0, :]

    heading_inv_rot = rotations.calc_heading_quat_inv(root_rot, w_last)

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    translation_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_current_state_body_pos.shape[1], 1)
    )
    rotation_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_current_state_body_rot.shape[1], 1)
    )
    flat_translation_heading_inv_rot = translation_heading_inv_rot_expand.reshape(
        translation_heading_inv_rot_expand.shape[0]
        * translation_heading_inv_rot_expand.shape[1],
        translation_heading_inv_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    flat_target_body_pos = (ref_state_body_pos - root_pos_expand).reshape(
        ref_state_body_pos.shape[0] * ref_state_body_pos.shape[1],
        ref_state_body_pos.shape[2],
    )
    flat_target_body_pos = rotations.quat_rotate(
        flat_translation_heading_inv_rot, flat_target_body_pos, w_last
    )
    target_body_pos = flat_target_body_pos.reshape(num_envs, future_steps, -1)

    flat_target_body_pos_rel = (
        ref_state_body_pos - flat_current_state_body_pos
    ).reshape(
        ref_state_body_pos.shape[0] * ref_state_body_pos.shape[1],
        ref_state_body_pos.shape[2],
    )
    flat_target_body_pos_rel = rotations.quat_rotate(
        flat_translation_heading_inv_rot, flat_target_body_pos_rel, w_last
    )
    target_body_pos_rel = flat_target_body_pos_rel.reshape(num_envs, future_steps, -1)

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        rotation_heading_inv_rot_expand, ref_state_body_rot, w_last
    )

    target_body_rot_obs = rotations.quat_to_tan_norm(
        target_body_rot.view(-1, 4), w_last
    ).reshape(num_envs, future_steps, -1)

    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_current_state_body_rot, w_last),
        ref_state_body_rot,
        w_last,
    )
    target_rel_body_rot_obs = rotations.quat_to_tan_norm(
        target_rel_body_rot.view(-1, 4), w_last
    ).reshape(num_envs, future_steps, -1)

    if with_relative:
        obs = torch.cat(
            (
                target_body_pos,
                target_body_pos_rel,
                target_body_rot_obs,
                target_rel_body_rot_obs,
            ),
            dim=-1,
        )
    else:
        obs = torch.cat(
            (
                target_body_pos,
                target_body_rot_obs,
            ),
            dim=-1,
        )

    if with_velocities:
        expanded_body_vel = current_state_body_vel.unsqueeze(1).expand(
            num_envs, future_steps, *current_state_body_vel.shape[1:]
        )
        flat_current_state_body_vel = expanded_body_vel.reshape(
            ref_state_body_vel.shape
        )

        flat_target_vel = (ref_state_body_vel - flat_current_state_body_vel).reshape(
            ref_state_body_vel.shape[0] * ref_state_body_vel.shape[1],
            ref_state_body_vel.shape[2],
        )
        flat_local_target_vel = rotations.quat_rotate(
            translation_heading_inv_rot_expand, flat_target_vel, w_last
        )
        local_target_vel = flat_local_target_vel.reshape(num_envs, future_steps, -1)

        expanded_body_ang_vel = current_state_body_ang_vel.unsqueeze(1).expand(
            num_envs, future_steps, *current_state_body_ang_vel.shape[1:]
        )
        flat_current_state_body_ang_vel = expanded_body_ang_vel.reshape(
            ref_state_body_ang_vel.shape
        )

        flat_target_body_ang_vel = (
            ref_state_body_ang_vel - flat_current_state_body_ang_vel
        ).reshape(
            ref_state_body_ang_vel.shape[0] * ref_state_body_ang_vel.shape[1],
            ref_state_body_ang_vel.shape[2],
        )
        flat_local_target_ang_vel = rotations.quat_rotate(
            rotation_heading_inv_rot_expand, flat_target_body_ang_vel, w_last
        )
        local_target_ang_vel = flat_local_target_ang_vel.reshape(
            num_envs, future_steps, -1
        )

        obs = torch.cat(
            (
                obs,
                local_target_vel,
                local_target_ang_vel,
            ),
            dim=-1,
        )

    return obs.view(num_envs, -1)


def build_reliability_gated_max_coords_target_poses_future_rel(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    mimic_ref_pos: Tensor,
    mimic_ref_rot: Tensor,
    future_reference_reliability: Tensor,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
    minimum_global_weight: float = 0.0,
    gravity_axis_only: bool = False,
    include_reliability: bool = True,
    include_trusted_vertical_anchors: bool = False,
    mimic_ref_vel: Optional[Tensor] = None,
    include_trusted_vertical_velocity: bool = False,
):
    """Build temporal articulation context without trusting noisy root jumps.

    The ordinary future-relative context exposes every observed shared root
    increment to the residual policy.  That silently bypasses reliability
    gating in the one-step command: a missing support can therefore command a
    spurious jump several frames in advance.  Here each root increment is
    accumulated with its own future confidence, while the same translation is
    applied to every body so root-relative articulation remains exact.

    At unit reliability this is exactly
    :func:`build_max_coords_target_poses_future_rel` (plus the optional
    confidence features).  With low vertical confidence, XY motion and all
    body-relative pose changes are preserved but the untrusted Z increment is
    withheld for contact dynamics to resolve.
    """
    if not 0.0 <= minimum_global_weight <= 1.0:
        raise ValueError("minimum_global_weight must lie in [0, 1]")
    if future_reference_reliability.ndim != 2:
        raise ValueError(
            "future_reference_reliability must have shape [envs, future_steps]"
        )
    if future_reference_reliability.shape[:2] != mimic_ref_pos.shape[:2]:
        raise ValueError(
            "future_reference_reliability must match the reference horizon"
        )
    if include_trusted_vertical_velocity:
        if mimic_ref_vel is None:
            raise ValueError(
                "mimic_ref_vel is required when "
                "include_trusted_vertical_velocity=True"
            )
        if mimic_ref_vel.shape[:2] != mimic_ref_pos.shape[:2]:
            raise ValueError("mimic_ref_vel must match the reference horizon")

    if future_steps is not None:
        mimic_ref_pos = select_step_indices(mimic_ref_pos, future_steps)
        mimic_ref_rot = select_step_indices(mimic_ref_rot, future_steps)
        future_reference_reliability = select_step_indices(
            future_reference_reliability.unsqueeze(-1), future_steps
        ).squeeze(-1)
        if mimic_ref_vel is not None:
            mimic_ref_vel = select_step_indices(mimic_ref_vel, future_steps)

    reliability = future_reference_reliability.clamp(0.0, 1.0)
    global_weight = minimum_global_weight + (
        1.0 - minimum_global_weight
    ) * reliability
    if gravity_axis_only:
        axis_weight = torch.ones(
            (*global_weight.shape, 3),
            dtype=global_weight.dtype,
            device=global_weight.device,
        )
        axis_weight[..., 2] = global_weight
    else:
        axis_weight = global_weight.unsqueeze(-1).expand(-1, -1, 3)

    reference_root = mimic_ref_pos[:, :, :1, :]
    previous_root = torch.cat(
        (current_state_body_pos[:, None, :1, :], reference_root[:, :-1]),
        dim=1,
    )
    root_increments = reference_root - previous_root
    gated_root = current_state_body_pos[:, None, :1, :] + torch.cumsum(
        root_increments * axis_weight[:, :, None, :], dim=1
    )
    gated_ref_pos = mimic_ref_pos + (gated_root - reference_root)

    context = build_max_coords_target_poses_future_rel(
        current_state_body_pos=current_state_body_pos,
        current_state_body_rot=current_state_body_rot,
        mimic_ref_pos=gated_ref_pos,
        mimic_ref_rot=mimic_ref_rot,
        w_last=w_last,
    )
    side_features = []
    if include_reliability:
        side_features.append(reliability)
    if include_trusted_vertical_anchors:
        # Unknown intermediate samples must not command their observed height,
        # but a later high-confidence support is useful as a planning anchor.
        # Expose that endpoint displacement separately from the smooth local
        # command, weighted by confidence so no noisy absolute Z leaks back in.
        trusted_vertical_anchor = reliability * (
            reference_root[:, :, 0, 2]
            - current_state_body_pos[:, None, 0, 2]
        )
        side_features.append(trusted_vertical_anchor)
    if include_trusted_vertical_velocity:
        # A future height alone does not tell the policy whether an upcoming
        # frame is a support, takeoff, or landing state.  The trusted world-Z
        # root velocity supplies that phase information early enough to prepare
        # an impulse.  It is yaw invariant, confidence gated, and remains a
        # command observation only: no action filtering or simulator force is
        # applied at deployment.
        trusted_vertical_velocity = reliability * mimic_ref_vel[:, :, 0, 2]
        side_features.append(trusted_vertical_velocity)
    if side_features:
        context = torch.cat((context, *side_features), dim=-1)
    return context


def build_reliability_gated_max_coords_target_poses(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    current_state_body_vel: Tensor,
    current_state_body_ang_vel: Tensor,
    mimic_ref_pos: Tensor,
    mimic_ref_rot: Tensor,
    mimic_ref_vel: Tensor,
    mimic_ref_ang_vel: Tensor,
    reference_reliability: Tensor,
    with_velocities: bool,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
    with_relative: bool = True,
    minimum_global_weight: float = 0.0,
    gravity_axis_only: bool = False,
):
    """Build the pretrained target encoding with confidence-gated root motion.

    A noisy global trajectory should not erase the observed articulation.  We
    decompose reference positions and linear velocities into a shared root
    component and root-relative body components.  Reliability attenuates only
    the shared translation and velocity before delegating to the exact
    pretrained max-coordinate encoder.  Body-relative geometry, orientations,
    angular velocities, feature order, and feature dimensionality are kept.

    At reliability one this is exactly :func:`build_max_coords_target_poses`.
    At reliability zero, the commanded root position and velocity coincide
    with the current simulated root.  Contacts and gravity can then determine
    global placement while articulation remains fully specified.
    """
    if not 0.0 <= minimum_global_weight <= 1.0:
        raise ValueError("minimum_global_weight must lie in [0, 1]")
    if reference_reliability.shape[0] != current_state_body_pos.shape[0]:
        raise ValueError("reference_reliability must have one value per environment")

    reliability = reference_reliability.reshape(
        reference_reliability.shape[0], -1
    )
    if reliability.shape[1] != 1:
        raise ValueError("reference_reliability must be scalar per environment")
    global_weight = minimum_global_weight + (
        1.0 - minimum_global_weight
    ) * reliability.clamp(0.0, 1.0)
    if gravity_axis_only:
        # Scene contact constrains the gravity-normal gauge most directly.
        # Preserve the observed horizontal trajectory while allowing physics
        # to reject floating/penetrating vertical root placement.
        axis_weight = torch.ones(
            (global_weight.shape[0], 3),
            dtype=global_weight.dtype,
            device=global_weight.device,
        )
        axis_weight[:, 2] = global_weight[:, 0]
        global_weight = axis_weight[:, None, None, :]
    else:
        global_weight = global_weight[:, None, None, :]

    current_root_pos = current_state_body_pos[:, None, :1, :]
    reference_root_pos = mimic_ref_pos[:, :, :1, :]
    gated_root_pos = current_root_pos + global_weight * (
        reference_root_pos - current_root_pos
    )
    gated_ref_pos = mimic_ref_pos + (gated_root_pos - reference_root_pos)

    current_root_vel = current_state_body_vel[:, None, :1, :]
    reference_root_vel = mimic_ref_vel[:, :, :1, :]
    gated_root_vel = current_root_vel + global_weight * (
        reference_root_vel - current_root_vel
    )
    gated_ref_vel = mimic_ref_vel + (gated_root_vel - reference_root_vel)

    return build_max_coords_target_poses(
        current_state_body_pos=current_state_body_pos,
        current_state_body_rot=current_state_body_rot,
        current_state_body_vel=current_state_body_vel,
        current_state_body_ang_vel=current_state_body_ang_vel,
        mimic_ref_pos=gated_ref_pos,
        mimic_ref_rot=mimic_ref_rot,
        mimic_ref_vel=gated_ref_vel,
        mimic_ref_ang_vel=mimic_ref_ang_vel,
        with_velocities=with_velocities,
        w_last=w_last,
        future_steps=future_steps,
        with_relative=with_relative,
    )


# Context mapping for ONNX export


def build_reduced_coords_target_poses(
    current_state_anchor_rot: Tensor,
    mimic_ref_anchor_rot: Tensor,
    mimic_ref_dof_vel: Tensor,
    mimic_ref_dof_pos: Tensor,
    w_last: bool = True,
    current_state_anchor_pos: Tensor = None,
    mimic_ref_anchor_pos: Tensor = None,
    mimic_ref_anchor_vel: Tensor = None,
    mimic_ref_anchor_ang_vel: Tensor = None,
    include_xy_offset: bool = False,
    include_height: bool = False,
    include_dof_vel: bool = True,
    include_anchor_vel: bool = False,
    include_anchor_ang_vel: bool = False,
    future_steps: Union[int, List[int]] = None,
    current_ref_anchor_pos: Tensor = None,
    zero_xy_offset: bool = False,
):
    """Build target pose observations in reduced coordinates.

    Args:
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_anchor_rot: Reference body rotations [envs, future_steps, 4]
        mimic_ref_dof_vel: Reference DOF velocities [envs, future_steps, num_dofs]
        mimic_ref_dof_pos: Reference DOF positions [envs, future_steps, num_dofs]
        w_last: If True, quaternions are in XYZW format
        current_state_anchor_pos: Current anchor position [envs, 3]
        mimic_ref_anchor_pos: Reference body positions [envs, future_steps, 3]
        mimic_ref_anchor_vel: Reference anchor linear velocity [envs, future_steps, 3]
        mimic_ref_anchor_ang_vel: Reference anchor angular velocity [envs, future_steps, 3]
        include_xy_offset: If True, includes XY offset [2]
        include_height: If True, includes absolute height [1]
        include_dof_vel: If True, includes DOF velocities [num_dofs]
        include_anchor_vel: If True, includes anchor linear velocity in local frame [3]
        include_anchor_ang_vel: If True, includes anchor angular velocity in local frame [3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        current_ref_anchor_pos: Current-frame reference anchor position [envs, 3].
            Used with include_xy_offset to compute drift from current reference.
        zero_xy_offset: If True, emit zeros for the XY offset (for inference).

    Returns:
        Target pose observations [envs, features]
    """
    num_envs = current_state_anchor_rot.shape[0]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_anchor_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)
        mimic_ref_dof_vel = select_step_indices(mimic_ref_dof_vel, future_steps)
        mimic_ref_dof_pos = select_step_indices(mimic_ref_dof_pos, future_steps)
        if mimic_ref_anchor_pos is not None:
            mimic_ref_anchor_pos = select_step_indices(
                mimic_ref_anchor_pos, future_steps
            )
        if mimic_ref_anchor_vel is not None:
            mimic_ref_anchor_vel = select_step_indices(
                mimic_ref_anchor_vel, future_steps
            )
        if mimic_ref_anchor_ang_vel is not None:
            mimic_ref_anchor_ang_vel = select_step_indices(
                mimic_ref_anchor_ang_vel, future_steps
            )

    future_steps = mimic_ref_anchor_rot.shape[1]

    # Flatten: [envs, future_steps, dim] -> [envs*future_steps, dim]
    ref_state_anchor_rot = mimic_ref_anchor_rot.reshape(-1, 4)
    ref_state_dof_vel = mimic_ref_dof_vel.reshape(-1, mimic_ref_dof_vel.shape[-1])
    ref_state_dof_pos = mimic_ref_dof_pos.reshape(-1, mimic_ref_dof_pos.shape[-1])

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)

    current_state_anchor_rot_expanded = (
        current_state_anchor_rot.unsqueeze(1)
        .expand(num_envs, future_steps, 4)
        .contiguous()
        .view(-1, 4)
    )

    heading_inv_rot_expanded = (
        heading_inv_rot.unsqueeze(1)
        .expand(num_envs, future_steps, 4)
        .contiguous()
        .view(-1, 4)
    )

    # Target root rot relative to current root rot
    rel_target_anchor_rot = rotations.quat_mul(
        rotations.quat_conjugate(current_state_anchor_rot_expanded, w_last),
        ref_state_anchor_rot,
        w_last,
    )
    target_anchor_rot_obs = rotations.quat_to_tan_norm(rel_target_anchor_rot, w_last)

    # Build observation components
    obs_components = [target_anchor_rot_obs]  # [N, 6]

    if include_dof_vel:
        obs_components.append(ref_state_dof_vel)  # [N, num_dofs]

    obs_components.append(ref_state_dof_pos)  # [N, num_dofs]

    if include_xy_offset or include_height:
        # Compute position components
        # Extract root position: [envs, future_steps, 3] -> [envs*future_steps, 3]
        ref_state_anchor_pos = mimic_ref_anchor_pos.reshape(-1, 3)

        if include_xy_offset:
            if zero_xy_offset:
                # Inference: zero out XY offset ("you're tracking perfectly")
                xy_offset_local = torch.zeros(
                    num_envs * future_steps,
                    2,
                    device=current_state_anchor_rot.device,
                    dtype=current_state_anchor_rot.dtype,
                )
            else:
                # Training: drift from current reference frame (state_t - hat_state_t)
                drift_origin = (
                    current_ref_anchor_pos
                    if current_ref_anchor_pos is not None
                    else current_state_anchor_pos
                )
                drift_origin_expanded = (
                    drift_origin.unsqueeze(1)
                    .expand(num_envs, future_steps, 3)
                    .contiguous()
                    .view(-1, 3)
                )

                # XY drift in world frame: agent pos - reference pos
                xy_drift_world = (
                    current_state_anchor_pos.unsqueeze(1)
                    .expand(num_envs, future_steps, 3)
                    .contiguous()
                    .view(-1, 3)[:, :2]
                    - drift_origin_expanded[:, :2]
                )

                # Rotate to heading-aligned frame
                xy_drift_3d = torch.cat(
                    [xy_drift_world, torch.zeros_like(xy_drift_world[:, :1])], dim=-1
                )
                xy_drift_local_3d = rotations.quat_rotate(
                    heading_inv_rot_expanded, xy_drift_3d, w_last
                )
                xy_offset_local = xy_drift_local_3d[:, :2]  # [num_envs*future_steps, 2]
            obs_components.append(xy_offset_local)

        if include_height:
            # Absolute height (not offset from current)
            height = ref_state_anchor_pos[:, 2:3]  # [num_envs*future_steps, 1]
            obs_components.append(height)

    if include_anchor_vel:
        if mimic_ref_anchor_vel is None:
            raise ValueError(
                "mimic_ref_anchor_vel is required when include_anchor_vel=True"
            )
        ref_state_anchor_vel = mimic_ref_anchor_vel.reshape(-1, 3)
        # Transform to local frame (heading-aligned)
        local_anchor_vel = rotations.quat_rotate(
            heading_inv_rot_expanded, ref_state_anchor_vel, w_last
        )
        obs_components.append(local_anchor_vel)

    if include_anchor_ang_vel:
        if mimic_ref_anchor_ang_vel is None:
            raise ValueError(
                "mimic_ref_anchor_ang_vel is required when include_anchor_ang_vel=True"
            )
        ref_state_anchor_ang_vel = mimic_ref_anchor_ang_vel.reshape(-1, 3)
        # Transform to local frame
        local_anchor_ang_vel = rotations.quat_rotate(
            heading_inv_rot_expanded, ref_state_anchor_ang_vel, w_last
        )
        obs_components.append(local_anchor_ang_vel)

    # Concatenate all observations
    obs = torch.cat(obs_components, dim=-1)

    # Reshape to [num_envs, future_steps * features]
    return obs.view(num_envs, -1)


# Context mapping for reduced coords target poses


def build_sparse_target_poses(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    masked_mimic_ref_pos: Tensor,
    masked_mimic_ref_rot: Tensor,
    conditionable_body_ids: Tensor,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
    include_root_relative: bool = True,
):
    """Build target pose observations for sparse body tracking (MaskedMimic).

    Similar to max_coords but only includes conditionable bodies (e.g., head, hands for VR).
    Provides both absolute and relative pose encodings for partial body control.

    Args:
        current_state_body_pos: Current body positions [envs, bodies, 3]
        current_state_body_rot: Current body rotations [envs, bodies, 4]
        masked_mimic_ref_pos: Target positions [envs, future_steps, bodies, 3]
        masked_mimic_ref_rot: Target rotations [envs, future_steps, bodies, 4]
        conditionable_body_ids: Indices of trackable bodies
        w_last: If True, quaternions are in XYZW format, else WXYZ
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        include_root_relative: If True (default), include both body-relative and root-relative
            poses (24 features per body). If False, only include body-relative poses
            (12 features per body: pos delta + rot delta from current to target).

    Returns:
        Sparse target pose observations [envs, features] for conditionable bodies only
    """
    num_envs = current_state_body_pos.shape[0]
    num_bodies = masked_mimic_ref_pos.shape[2]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        masked_mimic_ref_pos = select_step_indices(masked_mimic_ref_pos, future_steps)
        masked_mimic_ref_rot = select_step_indices(masked_mimic_ref_rot, future_steps)

    future_steps = masked_mimic_ref_pos.shape[1]

    # Flatten reference tensors: [envs, future_steps, bodies, dim] -> [envs*future_steps, bodies, dim]
    flat_target_body_pos = masked_mimic_ref_pos.reshape(-1, num_bodies, 3)
    flat_target_body_rot = masked_mimic_ref_rot.reshape(-1, num_bodies, 4)

    expanded_body_pos = current_state_body_pos.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_pos.shape[1:]
    )
    expanded_body_rot = current_state_body_rot.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_rot.shape[1:]
    )

    flat_cur_pos = expanded_body_pos.reshape(flat_target_body_pos.shape)
    flat_cur_rot = expanded_body_rot.reshape(flat_target_body_rot.shape)

    current_state_root_pos = flat_cur_pos[:, 0, :]
    current_state_root_rot = flat_cur_rot[:, 0, :]

    heading_rot = rotations.calc_heading_quat_inv(current_state_root_rot, w_last)

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, flat_cur_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    current_state_root_pos_expand = current_state_root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    target_rel_body_pos = flat_target_body_pos - flat_cur_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(
        target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
        target_rel_body_pos.shape[2],
    )
    flat_target_rel_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (
        flat_target_body_pos - current_state_root_pos_expand
    ).reshape(
        flat_target_body_pos.shape[0] * flat_target_body_pos.shape[1],
        flat_target_body_pos.shape[2],
    )
    flat_target_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_rot, w_last), flat_target_body_rot, w_last
    )
    target_rel_body_rot_obs = rotations.quat_to_tan_norm(
        target_rel_body_rot.view(-1, 4), w_last
    ).view(target_rel_body_rot.shape[0], -1)

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        heading_rot_expand, flat_target_body_rot, w_last
    )
    target_body_rot_obs = rotations.quat_to_tan_norm(
        target_body_rot.view(-1, 4), w_last
    ).view(target_rel_body_rot.shape[0], -1)

    padded_flat_target_rel_body_pos = torch.nn.functional.pad(
        flat_target_rel_body_pos, [0, 3], "constant", 0
    )
    sub_sampled_target_rel_body_pos = padded_flat_target_rel_body_pos.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]

    padded_flat_target_body_pos = torch.nn.functional.pad(
        flat_target_body_pos, [0, 3], "constant", 0
    )
    sub_sampled_target_body_pos = padded_flat_target_body_pos.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]

    sub_sampled_target_rel_body_rot_obs = target_rel_body_rot_obs.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]
    sub_sampled_target_body_rot_obs = target_body_rot_obs.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]

    # In masked_mimic allow easy re-shape to [batch, time, joint, type (transform/rotate), features]
    if include_root_relative:
        # Full output: body-relative + root-relative (24 features per body)
        obs = torch.cat(
            (
                sub_sampled_target_rel_body_pos,
                sub_sampled_target_body_pos,
                sub_sampled_target_rel_body_rot_obs,
                sub_sampled_target_body_rot_obs,
            ),
            dim=-1,  # [batch, timesteps, joints, 24]
        ).view(num_envs, -1)
    else:
        # Reduced output: only body-relative (12 features per body)
        # pos delta (current body -> target body) + rot delta (current rot -> target rot)
        obs = torch.cat(
            (
                sub_sampled_target_rel_body_pos,
                sub_sampled_target_rel_body_rot_obs,
            ),
            dim=-1,  # [batch, timesteps, joints, 12]
        ).view(num_envs, -1)

    return obs


# =============================================================================
# Individual Component Build Functions (for modular factories)
# =============================================================================
# All functions support future_steps for multi-frame targets.
# Output is flattened: [envs, future_steps * feature_dim]


def build_target_root_rot(
    current_state_root_rot: Tensor,
    mimic_ref_anchor_rot: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
) -> Tensor:
    """Build target root rotation observation (6D tan-norm).

    Args:
        current_state_root_rot: Current root rotation [envs, 4]
        mimic_ref_anchor_rot: Reference anchor rotation [envs, future_steps, 4]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format

    Returns:
        Relative root rotation [envs, future_steps * 6]
    """
    num_envs = current_state_root_rot.shape[0]

    # Slice to requested steps
    ref_anchor_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)
    actual_steps = ref_anchor_rot.shape[1]

    # Expand current rotation to match
    current_expanded = current_state_root_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    current_flat = current_expanded.reshape(-1, 4)  # [envs*steps, 4]
    ref_flat = ref_anchor_rot.reshape(-1, 4)  # [envs*steps, 4]

    # Relative rotation from current to target
    rel_target_root_rot = rotations.quat_mul(
        rotations.quat_conjugate(current_flat, w_last), ref_flat, w_last
    )
    rot_6d = rotations.quat_to_tan_norm(rel_target_root_rot, w_last)  # [envs*steps, 6]
    return rot_6d.reshape(num_envs, -1)


def build_target_xy_offset(
    current_state_anchor_pos: Tensor,
    current_state_anchor_rot: Tensor,
    mimic_ref_anchor_pos: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
    current_ref_anchor_pos: Tensor = None,
    zero_xy_offset: bool = False,
) -> Tensor:
    """Build target XY offset in heading frame.

    Computes drift from current reference frame (state_t - hat_state_t).

    Args:
        current_state_anchor_pos: Current anchor position [envs, 3]
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_anchor_pos: Reference anchor position [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format
        current_ref_anchor_pos: Current-frame reference anchor position [envs, 3].
            Used to compute drift from current reference.
        zero_xy_offset: If True, emit zeros (for inference).

    Returns:
        XY offset in heading frame [envs, future_steps * 2]
    """
    num_envs = current_state_anchor_pos.shape[0]

    # Slice to requested steps
    ref_anchor_pos = select_step_indices(mimic_ref_anchor_pos, future_steps)
    actual_steps = ref_anchor_pos.shape[1]

    if zero_xy_offset:
        return torch.zeros(
            num_envs,
            actual_steps * 2,
            device=current_state_anchor_pos.device,
            dtype=current_state_anchor_pos.dtype,
        )

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)
    heading_inv_expanded = heading_inv_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    heading_inv_flat = heading_inv_expanded.reshape(-1, 4)  # [envs*steps, 4]

    # Drift from current reference frame
    drift_origin = (
        current_ref_anchor_pos
        if current_ref_anchor_pos is not None
        else current_state_anchor_pos
    )
    drift_origin_expanded = drift_origin.unsqueeze(1).expand(-1, actual_steps, -1)

    # XY drift in world frame: agent pos - reference pos
    xy_drift_world = (
        current_state_anchor_pos.unsqueeze(1).expand(-1, actual_steps, -1)[:, :, :2]
        - drift_origin_expanded[:, :, :2]
    )

    # Rotate to heading-aligned frame
    xy_drift_flat = xy_drift_world.reshape(-1, 2)  # [envs*steps, 2]
    xy_drift_3d = torch.cat(
        [xy_drift_flat, torch.zeros_like(xy_drift_flat[:, :1])], dim=-1
    )
    xy_drift_local_3d = rotations.quat_rotate(heading_inv_flat, xy_drift_3d, w_last)
    xy_drift_local = xy_drift_local_3d[:, :2]  # [envs*steps, 2]

    return xy_drift_local.reshape(num_envs, -1)


def build_target_height(
    mimic_ref_anchor_pos: Tensor,
    future_steps: Union[int, List[int]] = 1,
) -> Tensor:
    """Build target absolute height observation.

    Args:
        mimic_ref_anchor_pos: Reference anchor position [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).

    Returns:
        Absolute height [envs, future_steps * 1]
    """
    num_envs = mimic_ref_anchor_pos.shape[0]
    ref_pos = select_step_indices(mimic_ref_anchor_pos, future_steps)
    heights = ref_pos[:, :, 2:3]  # [envs, steps, 1]
    return heights.reshape(num_envs, -1)


def build_target_root_vel(
    current_state_anchor_rot: Tensor,
    mimic_ref_root_vel: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
) -> Tensor:
    """Build target root linear velocity in local frame.

    Args:
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_root_vel: Reference root velocity [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format

    Returns:
        Root velocity in heading frame [envs, future_steps * 3]
    """
    num_envs = current_state_anchor_rot.shape[0]

    ref_root_vel = select_step_indices(mimic_ref_root_vel, future_steps)
    actual_steps = ref_root_vel.shape[1]

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)
    heading_inv_expanded = heading_inv_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    heading_inv_flat = heading_inv_expanded.reshape(-1, 4)  # [envs*steps, 4]

    ref_vel_flat = ref_root_vel.reshape(-1, 3)  # [envs*steps, 3]
    local_vel = rotations.quat_rotate(
        heading_inv_flat, ref_vel_flat, w_last
    )  # [envs*steps, 3]

    return local_vel.reshape(num_envs, -1)


def build_target_root_ang_vel(
    current_state_anchor_rot: Tensor,
    mimic_ref_root_ang_vel: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
) -> Tensor:
    """Build target root angular velocity in local frame.

    Args:
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_root_ang_vel: Reference root angular velocity [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format

    Returns:
        Root angular velocity in local frame [envs, future_steps * 3]
    """
    num_envs = current_state_anchor_rot.shape[0]

    ref_root_ang_vel = select_step_indices(mimic_ref_root_ang_vel, future_steps)
    actual_steps = ref_root_ang_vel.shape[1]

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)
    heading_inv_expanded = heading_inv_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    heading_inv_flat = heading_inv_expanded.reshape(-1, 4)  # [envs*steps, 4]

    ref_ang_vel_flat = ref_root_ang_vel.reshape(-1, 3)  # [envs*steps, 3]
    local_ang_vel = rotations.quat_rotate(
        heading_inv_flat, ref_ang_vel_flat, w_last
    )  # [envs*steps, 3]

    return local_ang_vel.reshape(num_envs, -1)


def build_deploy_target_poses(
    current_anchor_rot: Tensor,
    mimic_ref_rot: Tensor,
    mimic_ref_dof_pos: Tensor,
    mimic_ref_dof_vel: Tensor,
    w_last: bool = True,
    include_dof_vel: bool = True,
    future_steps: Union[int, List[int]] = None,
):
    """Build deployment-ready target pose observations.

    Only requires the robot's anchor/root orientation (from IMU during deployment)
    and reference motion data.  No position information is needed, making this
    suitable for deployment without external position tracking.

    The observation encodes:
    - Reference DOF positions (joint targets, frame-invariant)
    - Reference DOF velocities (joint velocity targets, frame-invariant)
    - Reference body rotations relative to current anchor orientation (6D per body)

    During deployment:
    1. At start, compute heading offset: heading(IMU) - heading(motion_root) at t=0
    2. Each step, apply heading offset to raw reference body rotations before feeding
       to this function.  In training, realign_motion_with_humanoid_on_each_step
       handles XY position alignment; rotation alignment at episode start is implicit
       from spawning at the reference pose.

    Args:
        current_anchor_rot: Current anchor body rotation [envs, 4] (IMU during deploy)
        mimic_ref_rot: Reference body rotations [envs, future_steps, num_bodies, 4]
        mimic_ref_dof_pos: Reference DOF positions [envs, future_steps, num_dofs]
        mimic_ref_dof_vel: Reference DOF velocities [envs, future_steps, num_dofs]
        w_last: If True, quaternions are in XYZW format
        include_dof_vel: If True, include DOF velocities in observation
        future_steps: Steps to select.  Int N for first N consecutive steps,
            list for specific step indices.  None = use all.

    Returns:
        Target pose observations [envs, features] containing:
        [ref_dof_pos, (ref_dof_vel), local_ref_body_rot_6d] per future step
    """
    num_envs = current_anchor_rot.shape[0]

    if future_steps is not None:
        mimic_ref_rot = select_step_indices(mimic_ref_rot, future_steps)
        mimic_ref_dof_pos = select_step_indices(mimic_ref_dof_pos, future_steps)
        mimic_ref_dof_vel = select_step_indices(mimic_ref_dof_vel, future_steps)

    n_future = mimic_ref_rot.shape[1]
    num_bodies = mimic_ref_rot.shape[2]

    ref_body_rot = mimic_ref_rot.reshape(-1, num_bodies, 4)
    ref_dof_pos = mimic_ref_dof_pos.reshape(-1, mimic_ref_dof_pos.shape[-1])
    ref_dof_vel = mimic_ref_dof_vel.reshape(-1, mimic_ref_dof_vel.shape[-1])

    current_rot_expanded = (
        current_anchor_rot.unsqueeze(1)
        .expand(num_envs, n_future, 4)
        .contiguous()
        .view(-1, 4)
    )
    current_rot_inv = rotations.quat_conjugate(current_rot_expanded, w_last)

    # ref body rotations in current anchor frame
    current_rot_inv_expand = current_rot_inv.unsqueeze(1).repeat(1, num_bodies, 1)
    local_ref_body_rot = rotations.quat_mul(
        current_rot_inv_expand, ref_body_rot, w_last
    )

    local_ref_body_rot_6d = rotations.quat_to_tan_norm(
        local_ref_body_rot.view(-1, 4), w_last
    ).reshape(num_envs * n_future, num_bodies * 6)

    obs_components = [ref_dof_pos]
    if include_dof_vel:
        obs_components.append(ref_dof_vel)
    obs_components.append(local_ref_body_rot_6d)

    obs = torch.cat(obs_components, dim=-1)
    return obs.view(num_envs, -1)
