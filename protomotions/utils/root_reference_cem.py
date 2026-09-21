# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Observation-anchored physics CEM for diagnosing noisy root translations.

The articulation is never changed.  CEM searches a low-frequency vertical
translation spline and uses hundreds of parallel simulator environments as the
objective.  A frame-wise *observation* reliability signal is converted to a
metric uncertainty and used as a Mahalanobis tether to the measured root path:
reliable visual evidence resists correction, while genuinely uncertain frames
can move farther.  This search is a diagnostic/reference proposal, not a
license to replace high-confidence image evidence with policy-preferred motion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RootReferenceCEMConfig:
    iterations: int = 12
    knots: int = 8
    elite_fraction: float = 0.10
    initial_std_m: float = 0.12
    minimum_std_m: float = 0.005
    update_momentum: float = 0.15
    minimum_correction_m: float = -0.60
    maximum_correction_m: float = 0.60
    failure_threshold_m: float = 0.50
    survival_weight: float = 5.0
    reward_weight: float = 1.0
    relative_pose_weight: float = 2.0
    root_tracking_weight: float = 1.0
    penetration_weight: float = 5.0
    contact_slide_weight: float = 0.20
    action_delta_weight: float = 0.03
    root_jerk_weight: float = 2.0e-5
    reliable_reference_weight: float = 0.25
    observation_sigma_min_m: float = 0.025
    observation_sigma_max_m: float = 0.20
    spline_curvature_weight: float = 0.25
    seed: int = 20260918

    def validate(self, population: int, frame_count: int) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.knots < 2 or self.knots > frame_count:
            raise ValueError("knots must lie in [2, frame_count]")
        if population < 4:
            raise ValueError("CEM needs at least four parallel candidates")
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError("elite_fraction must lie in (0, 1)")
        if max(2, int(round(population * self.elite_fraction))) >= population:
            raise ValueError("elite set must be smaller than the population")
        if self.minimum_std_m <= 0 or self.initial_std_m < self.minimum_std_m:
            raise ValueError("invalid CEM standard deviations")
        if not 0.0 <= self.update_momentum < 1.0:
            raise ValueError("update_momentum must lie in [0, 1)")
        if self.minimum_correction_m >= self.maximum_correction_m:
            raise ValueError("minimum correction must be below maximum correction")
        if not 0.0 < self.observation_sigma_min_m <= self.observation_sigma_max_m:
            raise ValueError("invalid observation uncertainty range")


def interpolate_root_knots(knots: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Linearly interpolate ``[population, knots]`` into frame trajectories."""
    if knots.ndim != 2:
        raise ValueError(f"knots must be rank two, got {tuple(knots.shape)}")
    if knots.shape[1] < 2:
        raise ValueError("at least two knots are required")
    if frame_count < 2:
        raise ValueError("at least two output frames are required")
    return F.interpolate(
        knots.unsqueeze(1),
        size=frame_count,
        mode="linear",
        align_corners=True,
    ).squeeze(1)


def fit_root_knots(trajectory: torch.Tensor, knot_count: int) -> torch.Tensor:
    """Sample a frame trajectory at evenly spaced knot locations."""
    trajectory = torch.as_tensor(trajectory)
    if trajectory.ndim != 1:
        raise ValueError("trajectory must be one-dimensional")
    if not 2 <= knot_count <= len(trajectory):
        raise ValueError("invalid knot_count")
    return F.interpolate(
        trajectory.view(1, 1, -1),
        size=knot_count,
        mode="linear",
        align_corners=True,
    ).view(-1)


def sample_cem_population(
    mean: torch.Tensor,
    std: torch.Tensor,
    population: int,
    lower: float,
    upper: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw antithetic candidates and always retain mean and zero baselines."""
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("mean and std must be same-shaped vectors")
    half = (population - 2 + 1) // 2
    noise = torch.randn(
        (half, len(mean)),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    stochastic = torch.cat((mean + std * noise, mean - std * noise), dim=0)
    candidates = torch.cat(
        (mean.unsqueeze(0), torch.zeros_like(mean).unsqueeze(0), stochastic),
        dim=0,
    )[:population]
    return candidates.clamp(lower, upper)


def update_cem_distribution(
    mean: torch.Tensor,
    std: torch.Tensor,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    elite_count: int,
    minimum_std: float,
    momentum: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Maximum-likelihood Gaussian update over the highest-scoring candidates."""
    if candidates.ndim != 2 or scores.shape != (len(candidates),):
        raise ValueError("candidate and score shapes do not match")
    elite_indices = torch.topk(scores, k=elite_count, largest=True).indices
    elite = candidates[elite_indices]
    elite_mean = elite.mean(dim=0)
    elite_std = elite.std(dim=0, unbiased=False).clamp_min(minimum_std)
    next_mean = momentum * mean + (1.0 - momentum) * elite_mean
    next_std = momentum * std + (1.0 - momentum) * elite_std
    return next_mean, next_std.clamp_min(minimum_std), elite_indices


def trajectory_regularization(
    frame_trajectory: torch.Tensor,
    knot_values: torch.Tensor,
    reliability: torch.Tensor,
    config: RootReferenceCEMConfig,
) -> Dict[str, torch.Tensor]:
    """Regularize corrections only where the observed reference is reliable."""
    if frame_trajectory.ndim != 2:
        raise ValueError("frame_trajectory must have shape [population, frames]")
    if reliability.shape != (frame_trajectory.shape[1],):
        raise ValueError("reliability must have one value per frame")
    reliability = reliability.clamp(0.0, 1.0)
    observation_sigma = config.observation_sigma_max_m - reliability * (
        config.observation_sigma_max_m - config.observation_sigma_min_m
    )
    reliable_reference = (
        frame_trajectory / observation_sigma.unsqueeze(0)
    ).square().mean(dim=1)
    if knot_values.shape[1] >= 3:
        curvature = (
            knot_values[:, :-2]
            - 2.0 * knot_values[:, 1:-1]
            + knot_values[:, 2:]
        ).square().mean(dim=1)
    else:
        curvature = torch.zeros_like(reliable_reference)
    total = (
        config.reliable_reference_weight * reliable_reference
        + config.spline_curvature_weight * curvature
    )
    return {
        "total": total,
        "reliable_reference": reliable_reference,
        "spline_curvature": curvature,
    }


def _load_prior(
    prior_motion: Optional[Path], frame_count: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    initial = torch.zeros(frame_count, dtype=torch.float32, device=device)
    reliability = torch.ones(frame_count, dtype=torch.float32, device=device)
    if prior_motion is None:
        return initial, reliability
    prior = torch.load(prior_motion, map_location="cpu", weights_only=False)
    if "root_translation_correction" in prior:
        correction = torch.as_tensor(prior["root_translation_correction"])
        if correction.shape != (frame_count, 3):
            raise ValueError("prior root correction has an incompatible shape")
        initial = correction[:, 2].to(device=device, dtype=torch.float32)
    reliability_key = (
        "root_observation_reliability"
        if "root_observation_reliability" in prior
        else "root_reference_reliability"
    )
    if reliability_key in prior:
        loaded_reliability = torch.as_tensor(prior[reliability_key])
        if loaded_reliability.shape != (frame_count,):
            raise ValueError("prior reliability has an incompatible shape")
        reliability = loaded_reliability.to(device=device, dtype=torch.float32)
    return initial, reliability.clamp(0.0, 1.0)


def _refresh_observations(agent: Any, env: Any):
    env._current_context = None
    env.compute_observations(context=env.context)
    observations = agent.add_agent_info_to_obs(env.get_obs())
    return agent.obs_dict_to_tensordict(observations)


@torch.no_grad()
def evaluate_root_candidates(
    agent: Any,
    env: Any,
    trajectories_z: torch.Tensor,
    reliability: torch.Tensor,
    config: RootReferenceCEMConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run one complete closed-loop rollout for every candidate in parallel."""
    population, frame_count = trajectories_z.shape
    if population != env.num_envs:
        raise ValueError("one simulator environment is required per CEM candidate")

    device = env.device
    trajectory_velocity_z = torch.empty_like(trajectories_z)
    trajectory_velocity_z[:, 1:-1] = (
        trajectories_z[:, 2:] - trajectories_z[:, :-2]
    ) / (2.0 * env.dt)
    trajectory_velocity_z[:, 0] = (
        trajectories_z[:, 1] - trajectories_z[:, 0]
    ) / env.dt
    trajectory_velocity_z[:, -1] = (
        trajectories_z[:, -1] - trajectories_z[:, -2]
    ) / env.dt
    env_ids = torch.arange(population, device=device, dtype=torch.long)
    env.motion_manager.motion_ids[env_ids] = 0
    env.motion_manager.motion_times[env_ids] = 0.0
    if hasattr(env.motion_manager, "set_clip_mode"):
        env.motion_manager.set_clip_mode(False)

    initial_residual = torch.zeros(population, 3, device=device)
    initial_residual[:, 2] = trajectories_z[:, 0]
    initial_velocity_residual = torch.zeros(population, 3, device=device)
    initial_velocity_residual[:, 2] = trajectory_velocity_z[:, 0]
    observations, _ = env.reset(
        env_ids,
        sample_flat=True,
        disable_motion_resample=True,
        reference_root_residual=initial_residual,
        reference_root_velocity_residual=initial_velocity_residual,
    )
    observations = agent.add_agent_info_to_obs(observations)
    observation_td = agent.obs_dict_to_tensordict(observations)

    totals = {
        "reward": torch.zeros(population, device=device),
        "relative_pose_error": torch.zeros(population, device=device),
        "root_tracking_error": torch.zeros(population, device=device),
        "penetration": torch.zeros(population, device=device),
        "contact_slide": torch.zeros(population, device=device),
        "action_delta": torch.zeros(population, device=device),
        "root_jerk": torch.zeros(population, device=device),
    }
    active = torch.ones(population, dtype=torch.bool, device=device)
    active_steps = torch.zeros(population, device=device)
    tracking_success_steps = torch.zeros(population, device=device)
    previous_action = None
    previous_root_velocity = None
    previous_root_acceleration = None

    body_names = env.robot_config.kinematic_info.body_names
    contact_ids = torch.tensor(
        [
            index
            for index, name in enumerate(body_names)
            if any(part in name.lower() for part in ("ankle", "toe", "foot"))
        ],
        dtype=torch.long,
        device=device,
    )

    for step in range(frame_count - 1):
        target_frame = step + 1
        residual = torch.zeros(population, 3, device=device)
        residual[:, 2] = trajectories_z[:, target_frame]
        velocity_residual = torch.zeros(population, 3, device=device)
        velocity_residual[:, 2] = trajectory_velocity_z[:, target_frame]
        env.set_reference_root_residual(residual, velocity_residual)
        observation_td = _refresh_observations(agent, env)
        model_output = agent.model(observation_td)
        actions = model_output.get("mean_action", model_output.get("action"))
        actions = torch.where(active.unsqueeze(-1), actions, torch.zeros_like(actions))

        _, rewards, _, terminated, _ = env.step(actions)
        state = env.simulator.get_robot_state()
        ref_state = env.motion_lib.get_motion_state(
            env.motion_manager.motion_ids,
            env.motion_manager.motion_times,
        )
        reference_positions = ref_state.rigid_body_pos.clone()
        reference_positions += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            reference_positions
        )
        reference_positions += residual[:, None, :]
        current_relative = state.rigid_body_pos - state.root_pos[:, None, :]
        reference_relative = (
            reference_positions - reference_positions[:, :1, :]
        )
        relative_pose_error = torch.linalg.vector_norm(
            current_relative - reference_relative, dim=-1
        ).mean(dim=-1)
        root_tracking_error = torch.linalg.vector_norm(
            state.root_pos - reference_positions[:, 0], dim=-1
        )
        absolute_tracking_error = torch.linalg.vector_norm(
            state.rigid_body_pos - reference_positions, dim=-1
        ).mean(dim=-1)

        ground = env.terrain.get_ground_heights(state.rigid_body_pos)
        center_penetration = (
            ground - state.rigid_body_pos[..., 2]
        ).clamp_min(0.0).mean(dim=-1)

        if contact_ids.numel() > 0:
            contact_force = torch.linalg.vector_norm(
                state.rigid_body_contact_forces[:, contact_ids], dim=-1
            )
            in_contact = contact_force > 20.0
            horizontal_speed = torch.linalg.vector_norm(
                state.rigid_body_vel[:, contact_ids, :2], dim=-1
            )
            contact_slide = (
                horizontal_speed * in_contact.float()
            ).sum(dim=-1) / in_contact.sum(dim=-1).clamp_min(1)
        else:
            contact_slide = torch.zeros(population, device=device)

        if previous_action is None:
            action_delta = torch.zeros(population, device=device)
        else:
            action_delta = torch.sqrt(
                (actions - previous_action).square().mean(dim=-1)
            )
        root_velocity = state.root_vel
        root_acceleration = (
            None
            if previous_root_velocity is None
            else (root_velocity - previous_root_velocity) / env.dt
        )
        if root_acceleration is None or previous_root_acceleration is None:
            root_jerk = torch.zeros(population, device=device)
        else:
            root_jerk = torch.linalg.vector_norm(
                (root_acceleration - previous_root_acceleration) / env.dt,
                dim=-1,
            )

        weight = active.float()
        totals["reward"] += rewards * weight
        # Robust caps prevent a body that has already fallen several metres
        # from numerically overwhelming all usable pre-failure evidence.  The
        # separate success fraction still records every >50 cm tracking miss.
        totals["relative_pose_error"] += relative_pose_error.clamp_max(1.0) * weight
        totals["root_tracking_error"] += root_tracking_error.clamp_max(2.0) * weight
        totals["penetration"] += center_penetration.clamp_max(0.25) * weight
        totals["contact_slide"] += contact_slide.clamp_max(3.0) * weight
        totals["action_delta"] += action_delta.clamp_max(1.0) * weight
        totals["root_jerk"] += root_jerk.clamp_max(2000.0) * weight
        active_steps += weight
        tracking_success_steps += (
            (absolute_tracking_error <= config.failure_threshold_m).float() * weight
        )

        finite_state = (
            torch.isfinite(state.root_pos).all(dim=-1)
            & torch.isfinite(state.root_vel).all(dim=-1)
            & torch.isfinite(rewards)
        )
        failed = terminated.bool() | ~finite_state
        active &= ~failed
        previous_action = actions
        previous_root_velocity = root_velocity
        previous_root_acceleration = root_acceleration

    denominator = active_steps.clamp_min(1.0)
    means = {name: value / denominator for name, value in totals.items()}
    survival = tracking_success_steps / float(frame_count - 1)
    score = (
        config.survival_weight * survival
        + config.reward_weight * means["reward"]
        - config.relative_pose_weight * means["relative_pose_error"]
        - config.root_tracking_weight * means["root_tracking_error"]
        - config.penetration_weight * means["penetration"]
        - config.contact_slide_weight * means["contact_slide"]
        - config.action_delta_weight * means["action_delta"]
        - config.root_jerk_weight * means["root_jerk"]
    )
    return score, {**means, "survival_fraction": survival}


def _statistics(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


@torch.no_grad()
def run_root_reference_cem(
    agent: Any,
    env: Any,
    source_motion_path: Path,
    output_dir: Path,
    config: RootReferenceCEMConfig,
    prior_motion_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Optimize, save and report one corrected single-motion reference."""
    source_motion_path = Path(source_motion_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source = torch.load(source_motion_path, map_location="cpu", weights_only=False)
    frame_counts = torch.as_tensor(source["motion_num_frames"], dtype=torch.long)
    if frame_counts.numel() != 1:
        raise ValueError("root-reference CEM currently requires exactly one motion")
    source_frame_count = int(frame_counts[0])
    population = env.num_envs
    source_dt = float(torch.as_tensor(source["motion_dt"])[0])
    motion_length = float(
        env.motion_lib.get_motion_length(
            torch.zeros(1, dtype=torch.long, device=env.device)
        )[0]
    )
    rollout_steps = max(1, int(motion_length / env.dt))
    rollout_frame_count = rollout_steps + 1
    config.validate(population, rollout_frame_count)

    prior_trajectory, source_reliability = _load_prior(
        None if prior_motion_path is None else Path(prior_motion_path),
        source_frame_count,
        env.device,
    )
    reliability = F.interpolate(
        source_reliability.view(1, 1, -1),
        size=rollout_frame_count,
        mode="linear",
        align_corners=True,
    ).view(-1)
    mean = fit_root_knots(prior_trajectory, config.knots)
    std = torch.full_like(mean, config.initial_std_m)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(config.seed)
    elite_count = max(2, int(round(population * config.elite_fraction)))

    best_score = -torch.inf
    best_knots = mean.clone()
    best_diagnostics: Dict[str, float] = {}
    history = []
    zero_baseline: Optional[Dict[str, Any]] = None
    agent.eval()

    for iteration in range(config.iterations):
        candidates = sample_cem_population(
            mean,
            std,
            population,
            config.minimum_correction_m,
            config.maximum_correction_m,
            generator,
        )
        trajectories = interpolate_root_knots(candidates, rollout_frame_count)
        rollout_score, diagnostics = evaluate_root_candidates(
            agent, env, trajectories, reliability, config
        )
        regularization = trajectory_regularization(
            trajectories, candidates, reliability, config
        )
        scores = rollout_score - regularization["total"]
        if zero_baseline is None:
            zero_index = 1
            zero_baseline = {
                "score": float(scores[zero_index]),
                **{
                    name: float(value[zero_index])
                    for name, value in diagnostics.items()
                },
                **{
                    f"regularization_{name}": float(value[zero_index])
                    for name, value in regularization.items()
                },
            }
        iteration_best_index = int(torch.argmax(scores))
        iteration_best_score = scores[iteration_best_index]
        if iteration_best_score > best_score:
            best_score = iteration_best_score.clone()
            best_knots = candidates[iteration_best_index].clone()
            best_diagnostics = {
                name: float(value[iteration_best_index])
                for name, value in diagnostics.items()
            }
            best_diagnostics.update(
                {
                    f"regularization_{name}": float(value[iteration_best_index])
                    for name, value in regularization.items()
                }
            )

        mean, std, elite_indices = update_cem_distribution(
            mean,
            std,
            candidates,
            scores,
            elite_count,
            config.minimum_std_m,
            config.update_momentum,
        )
        entry = {
            "iteration": iteration,
            "score": _statistics(scores),
            "elite_score_mean": float(scores[elite_indices].mean()),
            "best_score_so_far": float(best_score),
            "distribution_std_mean_m": float(std.mean()),
            "best_survival_fraction": best_diagnostics.get(
                "survival_fraction", 0.0
            ),
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)

    best_rollout_trajectory = interpolate_root_knots(
        best_knots.unsqueeze(0), rollout_frame_count
    )[0]
    best_source_trajectory = interpolate_root_knots(
        best_knots.unsqueeze(0), source_frame_count
    )[0]
    correction_xyz = torch.zeros(source_frame_count, 3, dtype=torch.float32)
    correction_xyz[:, 2] = best_source_trajectory.cpu()

    output = dict(source)
    source_positions = torch.as_tensor(source["gts"], dtype=torch.float32)
    corrected_positions = source_positions + correction_xyz[:, None, :]
    output["gts"] = corrected_positions
    # Keep the numerical CEM utilities importable in lightweight test
    # environments; pose_lib pulls in the complete MJCF stack.
    from protomotions.components.pose_lib import compute_cartesian_velocity

    output["gvs"] = compute_cartesian_velocity(
        corrected_positions,
        fps=1.0 / source_dt,
        velocity_max_horizon=3,
    ).to(torch.float32)
    output["root_translation_correction"] = correction_xyz
    output["root_reference_reliability"] = source_reliability.cpu()
    output["root_cem_metadata"] = {
        "method": "observation_anchored_parallel_physics_cem_v2",
        "config": asdict(config),
        "source_motion": str(source_motion_path),
        "prior_motion": None
        if prior_motion_path is None
        else str(Path(prior_motion_path).expanduser().resolve()),
        "best_score": float(best_score),
        "best_diagnostics": best_diagnostics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_motion = output_dir / "motionlib_root_cem.pt"
    torch.save(output, output_motion)
    report = {
        **output["root_cem_metadata"],
        "population": population,
        "source_frames": source_frame_count,
        "source_fps": 1.0 / source_dt,
        "simulation_steps": rollout_steps,
        "simulation_fps": 1.0 / env.dt,
        "elite_count": elite_count,
        "best_knots_z_m": [float(value) for value in best_knots.cpu()],
        "best_correction_z_m": _statistics(best_source_trajectory),
        "rollout_correction_z_m": _statistics(best_rollout_trajectory),
        "reliability": _statistics(source_reliability),
        "zero_correction_baseline": zero_baseline,
        "history": history,
        "output_motion": str(output_motion),
    }
    report_path = output_dir / "root_cem_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report
