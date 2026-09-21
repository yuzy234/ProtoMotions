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
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
from torch import Tensor
import math
from dataclasses import dataclass

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.components.motion_lib import MotionLib
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager


@dataclass
class MimicEpisodeContext:
    """Per-episode-batch state for mimic evaluation."""
    motion_ids: Tensor  # which motion each env is tracking
    frame_limits: Tensor  # how many frames before clip ends


class MimicEvaluator(BaseEvaluator):
    """Evaluator for Mimic agent's motion tracking performance."""

    def __init__(self, agent: Any, fabric: Any, config: MimicEvaluatorConfig):
        super().__init__(agent, fabric, config)

    @property
    def motion_lib(self) -> MotionLib:
        """Motion library (from agent)."""
        return self.agent.motion_lib

    @property
    def motion_manager(self) -> MimicMotionManager:
        """Motion manager (from env)."""
        return self.env.motion_manager

    def _register_plugins(self) -> None:
        """Register metric computation plugins."""
        self._register_smoothness_plugin(window_sec=0.4, high_jerk_threshold=6500.0)
        self._register_opening_jerk_plugin(duration_sec=0.5)
        self._register_action_smoothness_plugin()

    def _create_metrics(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, MotionMetrics]:
        """Create MotionMetrics buffers for trajectory collection (robot state + actions)."""
        metrics = {}

        self._add_robot_state_metrics(
            metrics, num_motions, motion_num_frames, max_eval_steps
        )

        num_dofs = self.env.robot_config.kinematic_info.num_dofs
        metrics["actions"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, num_dofs, device=self.device
        )

        return metrics

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking and cache env state for restoration."""
        num_motions = self.motion_lib.num_motions()
        motion_lengths = self.motion_lib.get_motion_length(None)
        motion_num_frames = (motion_lengths / self.env.dt).floor().long()
        motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
        self._init_eval_component_buffers(num_motions)

        # Cache env + motion manager state (restored in cleanup_after_evaluation)
        self._env_snapshot = self.env.save_state()
        self._cached_motion_ids = self.motion_manager.motion_ids.clone()
        self._cached_motion_times = self.motion_manager.motion_times.clone()
        self._cached_clip_mode = None
        if hasattr(self.motion_manager, "set_clip_mode"):
            self._cached_clip_mode = self.motion_manager.clip_mode_enabled
            # Periodic evaluation must run the complete reference motion, not
            # stop at the two-second training-window boundary.
            self.motion_manager.set_clip_mode(False)

        return self._create_metrics(
            num_motions, motion_num_frames, self.config.max_eval_steps
        )

    def _save_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """
        Save list of failed motions to a text file.

        Args:
            failed_motions: List of motion IDs that failed tracking
            epoch: Current epoch number
        """
        filename = f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        self._save_list_to_file(failed_motions, filename, subdirectory="failed_motions")

    def _update_motion_sampling_weights(self) -> None:
        """Update motion sampling weights based on evaluation component failures."""
        if self._motion_failed is None:
            return

        failed_motions = torch.nonzero(self._motion_failed).flatten().tolist()
        success_motions = torch.nonzero(~self._motion_failed).flatten().tolist()

        self._save_failed_motions(failed_motions, self.agent.current_epoch)

        success_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_success_discount,
            self.config.eval_metrics_every,
        )
        failure_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_failure_discount,
            self.config.eval_metrics_every,
        )
        new_weights = self.env.motion_manager.motion_weights.clone()
        new_weights[success_motions] *= success_discount
        if failure_discount != 0:
            new_weights[failed_motions] /= failure_discount
        else:
            new_weights[failed_motions] = 1.0
        self.env.motion_manager.update_sampling_weights(new_weights)

    def evaluate_episode(self, env_ids: torch.Tensor, max_steps: int) -> None:
        """Run a single episode batch, optionally with EMA action smoothing.

        When eval_action_ema_alpha is set, actions are low-pass filtered to
        simulate deployment conditions. Motions that fail under EMA get higher
        sampling weight, creating curriculum pressure toward smooth policies.
        """
        ema_alpha = self.config.eval_action_ema_alpha
        adaptive_alpha_min = self.config.eval_action_ema_alpha_min
        adaptive_alpha_max = self.config.eval_action_ema_alpha_max
        adaptive_enabled = (
            adaptive_alpha_min is not None and adaptive_alpha_max is not None
        )
        contact_alpha = self.config.eval_action_ema_contact_alpha
        if adaptive_enabled and adaptive_alpha_min > adaptive_alpha_max:
            raise ValueError(
                "eval_action_ema_alpha_min must not exceed "
                "eval_action_ema_alpha_max."
            )

        dof_names = self.env.robot_config.kinematic_info.dof_names
        body_names = self.env.robot_config.kinematic_info.body_names
        side_action_ids = {}
        side_contact_body_ids = {}
        for side in ("L", "R"):
            side_action_ids[side] = torch.tensor(
                [
                    index
                    for index, name in enumerate(dof_names)
                    if name.startswith(f"{side}_")
                    and any(
                        joint in name
                        for joint in ("Hip", "Knee", "Ankle", "Toe")
                    )
                ],
                dtype=torch.long,
                device=self.env.device,
            )
            side_contact_body_ids[side] = torch.tensor(
                [
                    index
                    for index, name in enumerate(body_names)
                    if name.startswith(f"{side}_")
                    and any(part in name for part in ("Ankle", "Toe", "Foot"))
                ],
                dtype=torch.long,
                device=self.env.device,
            )

        def smooth_actions(
            actions: torch.Tensor, previous: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if previous is None:
                return actions, actions.clone()
            if adaptive_enabled:
                delta_rms = torch.sqrt(
                    (actions - previous).square().mean(dim=-1, keepdim=True)
                )
                confidence = torch.exp(
                    -delta_rms / self.config.eval_action_ema_delta_scale
                )
                alpha = adaptive_alpha_min + (
                    adaptive_alpha_max - adaptive_alpha_min
                ) * confidence
            elif ema_alpha is not None:
                alpha = ema_alpha
            else:
                return actions, actions.clone()
            if not isinstance(alpha, torch.Tensor):
                alpha = actions.new_full((actions.shape[0], 1), float(alpha))
            alpha = alpha.expand_as(actions).clone()
            if contact_alpha is not None and actions.shape[-1] == len(dof_names):
                contact_forces = (
                    self.env.simulator.get_robot_state().rigid_body_contact_forces
                )
                for side in ("L", "R"):
                    action_ids = side_action_ids[side]
                    body_ids = side_contact_body_ids[side]
                    if action_ids.numel() == 0 or body_ids.numel() == 0:
                        continue
                    side_force = torch.linalg.vector_norm(
                        contact_forces[:, body_ids], dim=-1
                    ).amax(dim=-1)
                    in_contact = side_force > (
                        self.config.eval_action_ema_contact_force_threshold
                    )
                    capped = torch.minimum(
                        alpha[:, action_ids],
                        alpha.new_full(
                            (actions.shape[0], action_ids.numel()),
                            float(contact_alpha),
                        ),
                    )
                    alpha[:, action_ids] = torch.where(
                        in_contact.unsqueeze(-1),
                        capped,
                        alpha[:, action_ids],
                    )
            filtered = alpha * actions + (1.0 - alpha) * previous
            return filtered, filtered.clone()

        self._on_episode_start(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        prev_actions = None

        # A reference-state reset has no valid previous action/contact history.
        # Hold time at t=0 briefly so the policy can settle the terrain-safe
        # spawn before the motion and trajectory metrics begin.
        for _ in range(self.config.eval_pre_roll_steps):
            model_outs = self.agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            if ema_alpha is not None or adaptive_enabled:
                actions, prev_actions = smooth_actions(actions, prev_actions)
            self.env.step(actions)
            self.motion_manager.motion_times[env_ids] = 0.0
            # The step computed observations at t=dt. Rebuild them at the held
            # t=0 reference while preserving the newly established sim history.
            self.env._current_context = None
            self.env.compute_observations(context=self.env.context)
            obs = self.env.get_obs()
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

        # Pre-roll intentionally pins the public reference time to zero.  An
        # adaptive phase controller also carries a previous-rate prior, so
        # clear that private state before the scored rollout starts.
        adaptive_phase_enabled = False
        for component in self.env.control_manager.components.values():
            if getattr(component.config, "adaptive_phase_enabled", False):
                adaptive_phase_enabled = True
                component.reset_adaptive_phase_state(env_ids)

        # With adaptive phase, wall-clock step count is no longer the motion
        # frame index.  Score and record an environment only until its actual
        # reference clock reaches the clip end.  Continuing to the nominal
        # duration would compare a moving/falling robot against a frozen final
        # pose and can inflate errors by metres.
        active_mask = torch.ones(
            env_ids.shape[0], dtype=torch.bool, device=env_ids.device
        )

        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            if "root_pos_residual_scaled" in model_outs and hasattr(
                self.env, "set_reference_root_residual"
            ):
                self.env.set_reference_root_residual(
                    model_outs["root_pos_residual_scaled"]
                )

            # Apply EMA smoothing (deployment simulation)
            if ema_alpha is not None or adaptive_enabled:
                actions, prev_actions = smooth_actions(actions, prev_actions)

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

            self._episode_active_mask = active_mask.clone()
            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

            if adaptive_phase_enabled:
                reached_end = self.motion_manager.get_done_tracks(env_ids)
                active_mask &= ~reached_end
                if not active_mask.any():
                    break

    def run_evaluation(self) -> None:
        """Run evaluation across multiple motions."""
        for env_ids, motion_ids in self._build_eval_batches():
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            max_len = min(
                (motion_lengths.max() / self.env.dt).floor().long().item(),
                self.config.max_eval_steps,
            )
            # Build episode context before evaluate_episode so hooks can read it
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=(motion_lengths / self.env.dt).floor().long().clamp(
                    max=self.config.max_eval_steps
                ),
            )
            self.evaluate_episode(env_ids, max_len)

    def _build_eval_batches(self):
        """Build list of (env_ids, motion_ids) batches to evaluate.
        
        Returns:
            List of (env_ids, motion_ids) tuples
        """
        fixed_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )

        if fixed_motion_ids.numel() > 0:
            print(f"Only evaluating fixed motions: {fixed_motion_ids}")
            return [(first_env_indices, fixed_motion_ids)]

        num_motions = self.motion_lib.num_motions()
        batches = []
        for start in range(0, num_motions, self.num_envs):
            end = min(start + self.num_envs, num_motions)
            motion_ids = torch.arange(start, end, device=self.device)
            env_ids = torch.arange(0, motion_ids.numel(), device=self.device)
            print(f"Evaluating motions {start} to {end}, out of total {num_motions}")
            batches.append((env_ids, motion_ids))
        return batches

    # --- Hook overrides ---
    
    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Set motion_ids/times in the motion manager before reset."""
        self.motion_manager.motion_ids[env_ids] = self._episode_ctx.motion_ids
        self.motion_manager.motion_times[env_ids] = 0.0
    
    def _get_reset_kwargs(self) -> dict:
        """Customize env.reset() for mimic evaluation."""
        return {"sample_flat": True, "disable_motion_resample": True}
    
    def _check_eval_components(self, env_ids: Tensor, step_idx: int) -> None:
        """Filter by frame limits and check failures only for active clips."""
        still_active = self._episode_ctx.frame_limits > step_idx
        if hasattr(self, "_episode_active_mask"):
            still_active &= self._episode_active_mask
        if still_active.any():
            active_env_ids = env_ids[still_active]
            active_motion_ids = self._episode_ctx.motion_ids[still_active]
            self._check_evaluation_failures(active_env_ids, active_motion_ids)
    
    def _on_episode_step(self, env_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Collect smoothness metrics each step."""
        if hasattr(self, "_episode_active_mask"):
            active_mask = self._episode_active_mask
            env_ids = env_ids[active_mask]
            motion_ids = self._episode_ctx.motion_ids[active_mask]
        else:
            motion_ids = self._episode_ctx.motion_ids
        if env_ids.numel() == 0:
            return
        self._record_trajectory_step(
            self._metrics, extras, env_ids, motion_ids, actions
        )

    def _record_trajectory_step(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
        actions: Tensor,
    ) -> None:
        """Record robot state and actions into trajectory buffers for this step."""
        if "actions" in metrics and actions is not None:
            metrics["actions"].update(active_motion_ids, actions[active_env_ids].detach())

        for k in metrics.keys():
            if k == "actions":
                continue
            if f"raw/{k}" in extras:
                metrics[k].update(active_motion_ids, extras[f"raw/{k}"][active_env_ids].detach())

    def process_eval_results(self) -> Tuple[Dict, Optional[float]]:
        """Process results and update motion sampling weights."""
        to_log, success_rate = super().process_eval_results()
        self._update_motion_sampling_weights()

        additional_metrics = self._compute_additional_metrics(self._metrics)
        to_log.update(additional_metrics)

        evaluated_score = success_rate
        if getattr(self.config, "quality_checkpoint_score", False) and success_rate is not None:
            # Success remains the primary objective. Among controllers with the
            # same success rate, prefer lower tracking error and less temporal
            # jitter. These metrics are already produced by the full-motion
            # evaluator and therefore match deployment better than rollout loss.
            position_metric = getattr(
                self.config, "quality_score_position_metric", "gt_error"
            )
            position_metric_key = f"eval/{position_metric}/mean"
            if position_metric_key not in to_log:
                raise KeyError(
                    "Quality-score position metric is unavailable: "
                    f"{position_metric_key}"
                )
            penalties = {
                "gt": getattr(self.config, "quality_score_gt_weight", 1.0)
                * to_log[position_metric_key],
                "gr": getattr(self.config, "quality_score_gr_weight", 0.25)
                * to_log.get("eval/gr_error/mean", 0.0),
                "jerk": getattr(self.config, "quality_score_jerk_weight", 1.0e-4)
                * to_log.get("eval/normalized_jerk_mean", 0.0),
                "opening_jerk": getattr(
                    self.config, "quality_score_opening_jerk_weight", 2.0e-4
                )
                * to_log.get("eval/opening_body_jerk_rms_m_s3", 0.0),
                "action_delta": getattr(
                    self.config, "quality_score_action_delta_weight", 0.1
                )
                * to_log.get("eval/action_delta_mean_rad", 0.0),
            }
            evaluated_score = (
                getattr(self.config, "quality_score_success_weight", 10.0)
                * success_rate
                - sum(penalties.values())
            )
            to_log["eval/quality_score"] = evaluated_score
            for name, value in penalties.items():
                to_log[f"eval/quality_penalty/{name}"] = value

        if self.fabric.global_rank == 0:
            if (
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
            ):
                self._save_predicted_motion_lib(self._metrics, epoch=self.agent.current_epoch)

        return to_log, evaluated_score

    def cleanup_after_evaluation(self) -> None:
        """Restore env and motion manager state after evaluation."""
        self.motion_manager.motion_ids = self._cached_motion_ids
        self.motion_manager.motion_times = self._cached_motion_times
        if self._cached_clip_mode is not None:
            self.motion_manager.set_clip_mode(self._cached_clip_mode)
        self.env.restore_state(self._env_snapshot)
        
        del self._env_snapshot
        del self._cached_motion_ids
        del self._cached_motion_times
        del self._cached_clip_mode
        if hasattr(self, "_episode_active_mask"):
            del self._episode_active_mask
        super().cleanup_after_evaluation()

    def _plot_per_frame_metrics(
        self, metrics: Dict, actions_storage: list = None
    ) -> None:
        """
        Plot per-frame metrics vs time when evaluating a single motion.
        Uses base class plotting with custom colors for contact forces.

        Args:
            metrics: Dictionary of MotionMetrics objects
            actions_storage: List of action arrays for plotting (optional, currently unused)
        """
        # Define custom colors for specific metrics
        custom_colors = {}

        # Only plot metrics that were actually collected
        eval_metric_keys = list(self.config.evaluation_components.keys())
        available_keys = [k for k in eval_metric_keys if k in metrics]

        # Use base class generic plotting with custom colors
        super()._plot_per_frame_metrics(
            metrics,
            keys_to_plot=available_keys if available_keys else None,
            custom_colors=custom_colors,
            output_filename="metrics_per_frame_plot.png",
        )

    def _save_predicted_motion_lib(
        self, metrics: Dict[str, MotionMetrics], epoch: int
    ) -> None:
        """Pack collected predicted metrics and save as a MotionLib-compatible .pt file.

        This creates a predicted MotionLib and preserves per-motion identity/
        shape metadata from the source library. Local rotations are deliberately
        not copied because they would be inconsistent with the simulated pose.

        Args:
            metrics: Dictionary of MotionMetrics objects containing predicted data
            epoch: Current epoch number for filename
        """
        required_keys = [
            "dof_pos",
            "dof_vel",
            "rigid_body_pos",
            "rigid_body_rot",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "rigid_body_contacts",
        ]

        # Ensure required data exists
        for k in required_keys:
            if k not in metrics:
                raise ValueError(
                    f"Missing metric '{k}' required to build predicted MotionLib"
                )

        device = self.device
        num_motions = self.motion_lib.num_motions()

        motion_num_frames = metrics["dof_pos"].motion_lens.to(device=device).long()
        assert (
            motion_num_frames.shape[0] == num_motions
        ), "motion_num_frames size mismatch"

        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        motion_dt = (
            torch.ones(num_motions, dtype=torch.float32, device=device) * self.env.dt
        )
        motion_lengths = motion_num_frames.to(dtype=torch.float32) * self.env.dt

        def pack_metric(metric_key: str) -> torch.Tensor:
            data = metrics[metric_key].data
            per_motion = []
            for m in range(num_motions):
                f = motion_num_frames[m].item()
                f = min(f, data.shape[1])
                per_motion.append(data[m, :f].detach().clone())
            return torch.cat(per_motion, dim=0)

        # Build packed tensors matching MotionLib field names
        dps = pack_metric("dof_pos")  # [total_frames, num_dofs]
        dvs = pack_metric("dof_vel")  # [total_frames, num_dofs]

        # Rigid body tensors are stored flattened in metrics; reshape to [*, num_bodies, C]
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        gts_flat = pack_metric("rigid_body_pos")  # [total_frames, num_bodies*3]
        grs_flat = pack_metric("rigid_body_rot")  # [total_frames, num_bodies*4]
        gvs_flat = pack_metric("rigid_body_vel")  # [total_frames, num_bodies*3]
        gavs_flat = pack_metric("rigid_body_ang_vel")  # [total_frames, num_bodies*3]

        # Validate and reshape
        assert (
            gts_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_pos dim mismatch: {gts_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            grs_flat.shape[-1] == num_bodies * 4
        ), f"rigid_body_rot dim mismatch: {grs_flat.shape[-1]} vs {num_bodies*4}"
        assert (
            gvs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_vel dim mismatch: {gvs_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            gavs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_ang_vel dim mismatch: {gavs_flat.shape[-1]} vs {num_bodies*3}"

        gts = gts_flat.view(-1, num_bodies, 3)
        grs = grs_flat.view(-1, num_bodies, 4)
        gvs = gvs_flat.view(-1, num_bodies, 3)
        gavs = gavs_flat.view(-1, num_bodies, 3)

        # Pack predicted contacts from metrics
        contacts_data = metrics[
            "rigid_body_contacts"
        ].data  # [num_motions, max_frames, num_bodies]
        contacts_list = []
        for m in range(num_motions):
            f = motion_num_frames[m].item()
            # Clamp to available frames
            f = min(f, contacts_data.shape[1])
            # Convert float contacts to bool for consistency with MotionLib format
            contacts_list.append(contacts_data[m, :f].bool().detach().clone())
        contacts = torch.cat(contacts_list, dim=0)

        # Copy ground-truth motion weights and files
        gt_lib = self.motion_lib
        motion_weights = getattr(
            gt_lib,
            "motion_weights",
            torch.ones(num_motions, dtype=torch.float32, device=device),
        )
        motion_files = getattr(
            gt_lib,
            "motion_files",
            tuple([f"predicted_motion_{i}" for i in range(num_motions)]),
        )

        save_data = {
            "gts": gts,
            "grs": grs,
            "gvs": gvs,
            "gavs": gavs,
            "dvs": dvs,
            "dps": dps,
            "length_starts": length_starts,
            "motion_lengths": motion_lengths,
            "motion_dt": motion_dt,
            "motion_num_frames": motion_num_frames,
            "motion_weights": motion_weights,
            "motion_files": motion_files,
            "contacts": contacts,  # Always save predicted contacts
        }

        # These fields are motion-level metadata, not predicted state. Keeping
        # them is required for --smpl-shape-from-motion and traceability in a
        # physics-consistent-reference distillation stage.
        for key in (
            "motion_betas",
            "motion_genders",
            "motion_asset_files",
            "source_global_ids",
        ):
            value = getattr(gt_lib, key, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().clone()
            elif isinstance(value, (list, tuple)):
                value = tuple(value)
            save_data[key] = value

        # create dir if not exists
        output_dir = self.root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"predicted_motion_lib_epoch_{epoch}.pt"
        torch.save(save_data, output_path)
        print(f"Predicted MotionLib saved to {output_path}")
