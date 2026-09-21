# SPDX-License-Identifier: Apache-2.0
"""Robust latent residual with uncertainty-aware orientation tracking."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "latent_residual_robust_hybrid_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_latent_residual_reliability_base", _BASE_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
agent_config = base_experiment.agent_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        reliability_weighted_gr_rew_factory,
        reliability_weighted_relative_body_ori_rew_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    # The reset history contains zero actions, not the action needed to hold the
    # initialized pose. Do not penalize that artificial jump during grace.
    cfg.reward_components["action_smoothness"] = action_smoothness_factory(
        weight=-0.02,
        zero_during_grace_period=True,
    )
    cfg.reward_components["gr_rew"] = reliability_weighted_gr_rew_factory(
        weight=0.20,
        coefficient=-5.0,
        angular_speed_soft=4.0,
        angular_speed_scale=2.0,
        min_reliability=0.1,
    )
    cfg.reward_components["relative_body_ori_rew"] = (
        reliability_weighted_relative_body_ori_rew_factory(
            weight=0.10,
            sigma=0.40,
            angular_speed_soft=4.0,
            angular_speed_scale=2.0,
            min_reliability=0.1,
        )
    )
    # The velocity-only robust MotionLib remains useful, but a high angular
    # velocity reward can still make the controller chase residual frame noise.
    cfg.reward_components["gav_rew"].static_params["weight"] = 0.05
    return cfg
