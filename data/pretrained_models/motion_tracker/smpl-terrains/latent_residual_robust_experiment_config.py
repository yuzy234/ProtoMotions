# SPDX-License-Identifier: Apache-2.0
"""Latent residual for noisy references with a smooth safe-reset transition."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LATENT_EXPERIMENT_PATH = (
    Path(__file__).resolve().parent / "latent_residual_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_latent_residual_robust_base", _LATENT_EXPERIMENT_PATH
)
latent_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(latent_experiment)


terrain_config = latent_experiment.terrain_config
scene_lib_config = latent_experiment.scene_lib_config
motion_lib_config = latent_experiment.motion_lib_config
agent_config = latent_experiment.agent_config
configure_robot_and_simulator = latent_experiment.configure_robot_and_simulator
apply_inference_overrides = latent_experiment.apply_inference_overrides


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        blended_rh_rew_factory,
        blended_terrain_feasible_gt_rew_factory,
        reference_penetration_metric_factory,
    )
    from protomotions.envs.motion_manager.config import (
        OverlappingClipMotionManagerConfig,
    )

    cfg = latent_experiment.env_config(*args, **kwargs)
    cfg.safe_reference_reset_blend_time = 0.5
    # Fine-grained starts expose reset recovery at 0.1 s intervals while keeping
    # the same two-second rollout duration and failure-weighted curriculum.
    cfg.motion_manager = OverlappingClipMotionManagerConfig(
        clip_motion_id=0,
        clip_duration=2.0,
        clip_stride=0.1,
        failure_sampling_mix=0.8,
        max_failure_weight_ratio=4.0,
        init_start_prob=0.0,
        resample_on_reset=True,
    )
    cfg.reward_components["gt_rew"] = blended_terrain_feasible_gt_rew_factory(
        weight=0.5,
        coefficient=-25.0,
        penetration_margin=0.02,
        penetration_scale=0.05,
        min_z_weight=0.05,
    )
    cfg.reward_components["rh_rew"] = blended_rh_rew_factory(
        weight=0.1,
        coefficient=-20.0,
    )
    cfg.reward_components["reference_penetration_depth"] = (
        reference_penetration_metric_factory(fraction=False)
    )
    cfg.reward_components["reference_penetration_fraction"] = (
        reference_penetration_metric_factory(fraction=True)
    )
    return cfg
