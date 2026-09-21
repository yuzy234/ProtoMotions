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
"""Mimic control component for motion tracking tasks.

This component manages reference motion tracking, including:
- Motion manager for motion library sampling and playback
- Reference state computation and terrain correction
- Masked mimic conditioning state
- Visualization markers for target poses
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext, MimicContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.obs.humanoid import dof_to_local
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)
from protomotions.utils.rotations import quat_rotate_inverse

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class MimicControlConfig(ControlComponentConfig):
    """Configuration for mimic control component.
    
    Attributes:
        bootstrap_on_episode_end: If True, don't terminate when motion clip ends.
        future_steps: Future reference poses to provide in context. If int N,
            provides N consecutive steps (1 to N). If list, provides specific step
            indices (e.g., [1, 3, 5, 9, 15] for non-uniform sampling).
    """
    _target_: str = "protomotions.envs.control.mimic_control.MimicControl"
    
    bootstrap_on_episode_end: bool = True
    future_steps: Union[int, List[int]] = 1
    # Motion confidence is an uncertainty estimate, not necessarily the best
    # linear control gain.  The optional calibrated gate preserves the exact
    # pretrained command above ``high`` and only relaxes world-space tracking
    # in genuinely ambiguous frames.  Defaults retain legacy linear behavior.
    reference_reliability_gate_low: float = 0.0
    reference_reliability_gate_high: float = 1.0
    reference_reliability_gate_smoothstep: bool = False
    # Optional monotone online time warping.  It is disabled for every legacy
    # experiment.  When enabled, reliable contact phases remain at 1x while
    # physically inconsistent video phases may advance faster according to
    # current articulated-state agreement.
    adaptive_phase_enabled: bool = False
    adaptive_phase_rates: List[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0]
    )
    adaptive_phase_unreliable_below: float = 0.25
    adaptive_phase_reliable_above: float = 0.75
    adaptive_phase_contact_lock_reliability: float = 0.80
    adaptive_phase_pose_scale: float = 0.20
    adaptive_phase_prior_weight: float = 0.25
    adaptive_phase_change_weight: float = 0.05


def calibrate_reference_reliability(
    confidence: Tensor,
    low: float = 0.0,
    high: float = 1.0,
    smoothstep: bool = False,
) -> Tensor:
    """Map evidence confidence to an absolute-tracking control weight.

    A confidence around 0.7 can still be strong evidence, so treating it as a
    literal 30% reduction of the root command causes drift in otherwise clean
    clips.  A clipped smoothstep gives the controller an explicit trusted
    plateau while remaining continuous through uncertain frames.
    """
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("reliability gate requires 0 <= low < high <= 1")
    confidence = confidence.float().clamp(0.0, 1.0)
    if not smoothstep:
        return confidence
    normalized = ((confidence - low) / (high - low)).clamp(0.0, 1.0)
    return normalized.square() * (3.0 - 2.0 * normalized)


def select_reliability_adaptive_phase_rate(
    current_local_pos: Tensor,
    candidate_local_pos: Tensor,
    candidate_rates: Tensor,
    physical_reliability: Tensor,
    previous_rate: Tensor,
    *,
    unreliable_below: float,
    reliable_above: float,
    contact_lock_reliability: float,
    pose_scale: float,
    prior_weight: float,
    change_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Select monotone phase speed from physics prior and state agreement.

    The physics term supplies direction only where source timing is unreliable;
    the pose term prevents blind skipping when the simulated articulation is not
    ready.  High-confidence support/contact frames are hard anchors at 1x.
    """
    if current_local_pos.ndim != 3 or candidate_local_pos.ndim != 4:
        raise ValueError("expected current [E,B,3] and candidates [E,C,B,3]")
    if candidate_local_pos.shape[0] != current_local_pos.shape[0]:
        raise ValueError("current and candidate environment counts differ")
    if candidate_local_pos.shape[2:] != current_local_pos.shape[1:]:
        raise ValueError("current and candidate body shapes differ")
    rates = torch.as_tensor(
        candidate_rates,
        dtype=current_local_pos.dtype,
        device=current_local_pos.device,
    ).flatten()
    if rates.numel() != candidate_local_pos.shape[1]:
        raise ValueError("candidate rate count does not match candidate poses")
    if torch.any(rates < 1.0) or not torch.any(torch.isclose(rates, rates.new_tensor(1.0))):
        raise ValueError("adaptive phase rates must be >= 1 and include 1x")
    if not 0.0 <= unreliable_below < reliable_above <= 1.0:
        raise ValueError("adaptive phase reliability thresholds are invalid")
    if not 0.0 <= contact_lock_reliability <= 1.0:
        raise ValueError("contact lock reliability must lie in [0, 1]")
    if pose_scale <= 0 or prior_weight < 0 or change_weight < 0:
        raise ValueError("adaptive phase cost scales must be non-negative")

    pose_cost = torch.mean(
        torch.square(
            candidate_local_pos - current_local_pos.unsqueeze(1)
        ),
        dim=(-1, -2),
    ) / (pose_scale * pose_scale)
    physical_reliability = physical_reliability.float().clamp(0.0, 1.0)
    unreliable = (
        (reliable_above - physical_reliability)
        / (reliable_above - unreliable_below)
    ).clamp(0.0, 1.0)
    unreliable = unreliable.square() * (3.0 - 2.0 * unreliable)
    preferred_rate = 1.0 + (rates.max() - 1.0) * unreliable
    total_cost = pose_cost
    total_cost = total_cost + prior_weight * torch.square(
        rates.unsqueeze(0) - preferred_rate.unsqueeze(1)
    )
    total_cost = total_cost + change_weight * torch.square(
        rates.unsqueeze(0) - previous_rate.unsqueeze(1)
    )

    rate_one = torch.isclose(rates, rates.new_tensor(1.0))
    locked = physical_reliability >= contact_lock_reliability
    total_cost = torch.where(
        locked.unsqueeze(1) & (~rate_one.unsqueeze(0)),
        torch.full_like(total_cost, torch.inf),
        total_cost,
    )
    selected_indices = torch.argmin(total_cost, dim=1)
    selected_rate = rates[selected_indices]
    selected_pose_cost = pose_cost.gather(1, selected_indices[:, None]).squeeze(1)
    return selected_rate, preferred_rate, selected_pose_cost


class MimicControl(ControlComponent):
    """Control component for motion tracking tasks.
    
    Provides context for mimic observations and rewards. Accesses the env's
    motion_manager rather than creating its own.
    """
    
    config: MimicControlConfig
    
    def __init__(self, config: MimicControlConfig, env: "BaseEnv"):
        """Initialize mimic control component.
        
        Args:
            config: Component configuration.
            env: Parent environment instance.
        """
        super().__init__(config, env)
        rates = torch.as_tensor(
            getattr(config, "adaptive_phase_rates", [1.0, 1.5, 2.0, 2.5, 3.0]),
            dtype=torch.float32,
            device=env.device,
        )
        if rates.ndim != 1 or rates.numel() == 0:
            raise ValueError("adaptive_phase_rates must be a non-empty list")
        if torch.any(rates < 1.0) or not torch.any(torch.isclose(rates, rates.new_tensor(1.0))):
            raise ValueError("adaptive_phase_rates must be >= 1 and include 1x")
        self._adaptive_phase_rates = rates
        self._adaptive_phase_rate = torch.ones(
            env.num_envs, dtype=torch.float32, device=env.device
        )

    @staticmethod
    def _root_local_body_positions(state) -> Tensor:
        positions = state.rigid_body_pos
        rotations = state.rigid_body_rot
        relative = positions - positions[..., :1, :]
        root_rotation = rotations[..., :1, :].expand(relative.shape[:-1] + (4,))
        local = quat_rotate_inverse(
            root_rotation.reshape(-1, 4),
            relative.reshape(-1, 3),
            True,
        )
        return local.reshape_as(relative)

    def reset(self, env_ids: Tensor):
        self.reset_adaptive_phase_state(env_ids)

    def reset_adaptive_phase_state(self, env_ids: Tensor) -> None:
        """Reset the previous-rate prior without changing the reference pose."""
        self._adaptive_phase_rate[env_ids] = 1.0
    
    def step(self):
        """Control component step - motion manager is handled by env."""
        if not getattr(self.config, "adaptive_phase_enabled", False):
            return

        manager = self.env.motion_manager
        motion_ids = manager.motion_ids
        nominal_times = manager.motion_times
        rates = self._adaptive_phase_rates
        candidate_times = nominal_times.unsqueeze(1) + (
            rates.unsqueeze(0) - 1.0
        ) * self.env.dt
        motion_lengths = self.env.motion_lib.get_motion_length(motion_ids)
        candidate_times = torch.minimum(
            candidate_times, motion_lengths.unsqueeze(1)
        )
        if getattr(manager, "clip_mode_enabled", False) and hasattr(
            manager, "current_clip_end_times"
        ):
            candidate_times = torch.minimum(
                candidate_times, manager.current_clip_end_times.unsqueeze(1)
            )

        candidate_count = rates.numel()
        candidate_state = self.env.motion_lib.get_motion_state(
            motion_ids.unsqueeze(1).expand(-1, candidate_count).reshape(-1),
            candidate_times.reshape(-1),
        )
        num_bodies = candidate_state.rigid_body_pos.shape[1]
        candidate_state.rigid_body_pos = candidate_state.rigid_body_pos.reshape(
            self.env.num_envs, candidate_count, num_bodies, 3
        )
        candidate_state.rigid_body_rot = candidate_state.rigid_body_rot.reshape(
            self.env.num_envs, candidate_count, num_bodies, 4
        )
        candidate_local = self._root_local_body_positions(candidate_state)
        current_local = self._root_local_body_positions(
            self.env.simulator.get_robot_state()
        )
        physical_reliability = (
            self.env.motion_lib.get_root_physical_reliability(
                motion_ids, nominal_times
            )
        )
        selected_rate, preferred_rate, selected_pose_cost = (
            select_reliability_adaptive_phase_rate(
                current_local,
                candidate_local,
                rates,
                physical_reliability,
                self._adaptive_phase_rate,
                unreliable_below=self.config.adaptive_phase_unreliable_below,
                reliable_above=self.config.adaptive_phase_reliable_above,
                contact_lock_reliability=(
                    self.config.adaptive_phase_contact_lock_reliability
                ),
                pose_scale=self.config.adaptive_phase_pose_scale,
                prior_weight=self.config.adaptive_phase_prior_weight,
                change_weight=self.config.adaptive_phase_change_weight,
            )
        )
        selected_times = candidate_times.gather(
            1,
            torch.argmin(
                torch.abs(rates.unsqueeze(0) - selected_rate.unsqueeze(1)), dim=1
            )[:, None],
        ).squeeze(1)
        manager.motion_times.copy_(selected_times)
        self._adaptive_phase_rate.copy_(selected_rate)
        self.env.extras["phase/selected_rate_mean"] = selected_rate.mean()
        self.env.extras["phase/preferred_rate_mean"] = preferred_rate.mean()
        self.env.extras["phase/pose_cost_mean"] = selected_pose_cost.mean()
        self.env.extras["phase/physical_reliability_mean"] = (
            physical_reliability.mean()
        )
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check if motion clips have finished.
        
        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors.
        """
        device = self.env.device
        num_envs = self.env.num_envs
        
        # Check if motion clip has finished (access via env)
        done_clip = self.env.motion_manager.get_done_tracks()
        reset_buf = done_clip
        
        # Only terminate if not bootstrapping
        if self.config.bootstrap_on_episode_end:
            terminate_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        else:
            terminate_buf = done_clip
        
        return reset_buf, terminate_buf
    
    def populate_context(self, ctx: EnvContext) -> None:
        """Populate mimic-specific view in the EnvContext.
        
        Creates a MimicContext with:
        - ref_state: Single-step reference state at current time (for rewards)
        - future_*: Multi-step future reference poses [envs, future_steps, ...]
        
        Args:
            ctx: The EnvContext to populate with ctx.mimic.
        """
        num_envs = self.env.num_envs
        device = self.env.device
        motion_ids = self.env.motion_manager.motion_ids
        motion_times = self.env.motion_manager.motion_times
        reference_reliability = (
            self.env.motion_lib.get_root_reference_reliability(
                motion_ids, motion_times
            )
        )
        reference_reliability = calibrate_reference_reliability(
            reference_reliability,
            low=self.config.reference_reliability_gate_low,
            high=self.config.reference_reliability_gate_high,
            smoothstep=self.config.reference_reliability_gate_smoothstep,
        )
        
        # Get single-step reference state at current time (for rewards)
        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)
        
        # Apply terrain height correction to reference state
        ref_gt = ref_state.rigid_body_pos.clone()
        ref_gt += self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_gt
        )
        if hasattr(self.env, "reference_root_residual"):
            ref_gt += self.env.reference_root_residual[:, None, :]
        ref_state.rigid_body_pos = ref_gt
        if hasattr(self.env, "reference_root_velocity_residual"):
            ref_state.rigid_body_vel = ref_state.rigid_body_vel.clone()
            ref_state.rigid_body_vel += (
                self.env.reference_root_velocity_residual[:, None, :]
            )
        
        # Build multi-step reference for observations
        dt = self.env.dt
        if isinstance(self.config.future_steps, int):
            step_indices = list(range(1, self.config.future_steps + 1))
        else:
            step_indices = self.config.future_steps
        future_steps = len(step_indices)
        
        time_offsets = dt * torch.tensor(step_indices, device=device, dtype=torch.float32)
        future_times = motion_times.unsqueeze(-1) + time_offsets  # [envs, N]
        
        motion_lengths = self.env.motion_lib.get_motion_length(motion_ids)
        future_times = torch.minimum(future_times, motion_lengths.unsqueeze(-1))
        
        flat_motion_ids = motion_ids.unsqueeze(-1).expand(
            num_envs, future_steps
        ).reshape(-1)
        flat_future_times = future_times.reshape(-1)

        future_reference_reliability = (
            self.env.motion_lib.get_root_reference_reliability(
                flat_motion_ids, flat_future_times
            )
            .view(num_envs, future_steps)
        )
        future_reference_reliability = calibrate_reference_reliability(
            future_reference_reliability,
            low=self.config.reference_reliability_gate_low,
            high=self.config.reference_reliability_gate_high,
            smoothstep=self.config.reference_reliability_gate_smoothstep,
        )
        
        # Query motion lib for all future steps
        future_state = self.env.motion_lib.get_motion_state(
            flat_motion_ids, flat_future_times
        )
        
        # Reshape to [envs, future_steps, ...] and apply terrain correction
        num_bodies = future_state.rigid_body_pos.shape[1]
        num_dofs = future_state.dof_pos.shape[1]
        
        # Body positions with terrain correction
        future_pos = future_state.rigid_body_pos.view(
            num_envs, future_steps, num_bodies, 3
        ).clone()
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            future_pos[:, 0, :, :]  # Use first step for offset
        )
        future_pos += offset.unsqueeze(1)
        if hasattr(self.env, "reference_root_residual"):
            future_pos += self.env.reference_root_residual[:, None, None, :]
        
        # Body rotations
        future_rot = future_state.rigid_body_rot.view(
            num_envs, future_steps, num_bodies, 4
        )
        
        # Body velocities
        future_vel = future_state.rigid_body_vel.view(
            num_envs, future_steps, num_bodies, 3
        )
        if hasattr(self.env, "reference_root_velocity_residual"):
            future_vel = future_vel.clone()
            future_vel += self.env.reference_root_velocity_residual[
                :, None, None, :
            ]
        
        # Body angular velocities
        future_ang_vel = future_state.rigid_body_ang_vel.view(
            num_envs, future_steps, num_bodies, 3
        )
        
        # DOF positions and velocities
        future_dof_pos = future_state.dof_pos.view(
            num_envs, future_steps, num_dofs
        )
        future_dof_vel = future_state.dof_vel.view(
            num_envs, future_steps, num_dofs
        )
        
        hinge_axes_map = self.env.robot_config.kinematic_info.hinge_axes_map
        ref_lr = dof_to_local(ref_state.dof_pos, hinge_axes_map, True)
        ref_ground_heights = self.env.terrain.get_ground_heights(
            ref_state.rigid_body_pos
        )
        blend_time = self.env.config.safe_reference_reset_blend_time
        if blend_time > 0.0:
            blend_progress = (
                self.env.progress_buf.float() * self.env.dt / blend_time
            ).clamp(0.0, 1.0)
            # 1 - smoothstep(0, 1): zero endpoint slope avoids a target-velocity jump.
            blend_factor = 1.0 - blend_progress.pow(2) * (3.0 - 2.0 * blend_progress)
            reset_transition_progress = 1.0 - blend_factor
            safe_reset_reward_lift = (
                self.env.safe_reference_reset_lift * blend_factor
            )
        else:
            safe_reset_reward_lift = torch.zeros_like(
                self.env.safe_reference_reset_lift
            )
            reset_transition_progress = torch.ones_like(
                self.env.safe_reference_reset_lift
            )
        
        # Populate the mimic view
        ctx.mimic = MimicContext(
            ref_state=ref_state,
            future_pos=future_pos,
            future_rot=future_rot,
            future_vel=future_vel,
            future_ang_vel=future_ang_vel,
            future_dof_pos=future_dof_pos,
            future_dof_vel=future_dof_vel,
            anchor_idx=self.env.robot_config.anchor_body_index,
            ref_lr=ref_lr,
            ref_ground_heights=ref_ground_heights,
            safe_reset_reward_lift=safe_reset_reward_lift,
            reset_transition_progress=reset_transition_progress,
            reference_reliability=reference_reliability,
            future_reference_reliability=future_reference_reliability,
        )
    
    def create_visualization_markers(self, headless: bool) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for reference poses.
        
        Args:
            headless: If True, returns empty dict.
            
        Returns:
            Dictionary of marker configurations.
        """
        if headless:
            return {}
        
        visualization_markers = {}
        
        # Standard mimic: visualize all bodies
        body_names = self.env.robot_config.kinematic_info.body_names
        
        body_markers = []
        for body_name in body_names:
            if (
                self.env.robot_config.mimic_small_marker_bodies is not None
                and body_name in self.env.robot_config.mimic_small_marker_bodies
            ):
                body_markers.append(MarkerConfig(size="small"))
            else:
                body_markers.append(MarkerConfig(size="regular"))
        
        # Red markers for target poses
        body_markers_red_cfg = VisualizationMarkerConfig(
            type="sphere", color=(1.0, 0.0, 0.0), markers=body_markers
        )
        visualization_markers["body_markers_red"] = body_markers_red_cfg
        
        return visualization_markers
    
    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Compute marker positions for reference poses.
        
        Returns:
            Dictionary mapping marker names to MarkerState.
        """
        if self.env.simulator.headless:
            return {}
        
        markers_state = {}
        
        # Get reference state at current time (access motion_manager via env)
        ref_state = self.env.motion_lib.get_motion_state(
            self.env.motion_manager.motion_ids, self.env.motion_manager.motion_times
        )
        
        target_pos = ref_state.rigid_body_pos.clone()
        target_pos += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                target_pos
            )
        )
        if hasattr(self.env, "reference_root_residual"):
            target_pos += self.env.reference_root_residual[:, None, :]
        
        # Standard mimic: show all body markers in red
        target_pos = target_pos.view(self.env.num_envs, -1, 3)
        markers_state["body_markers_red"] = MarkerState(
            translation=target_pos,
            orientation=torch.zeros(
                self.env.num_envs, target_pos.shape[1], 4, device=self.env.device
            ),
        )
        
        return markers_state
