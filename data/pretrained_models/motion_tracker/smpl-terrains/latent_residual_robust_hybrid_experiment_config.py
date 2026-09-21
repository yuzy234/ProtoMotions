# SPDX-License-Identifier: Apache-2.0
"""Robust noisy-reference tracker with hybrid pose and foot-slip objectives."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ROBUST_PATH = (
    Path(__file__).resolve().parent / "latent_residual_robust_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_latent_residual_robust_hybrid_base", _ROBUST_PATH
)
robust_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(robust_experiment)


terrain_config = robust_experiment.terrain_config
scene_lib_config = robust_experiment.scene_lib_config
motion_lib_config = robust_experiment.motion_lib_config
configure_robot_and_simulator = robust_experiment.configure_robot_and_simulator
apply_inference_overrides = robust_experiment.apply_inference_overrides


def agent_config(*args, **kwargs):
    cfg = robust_experiment.agent_config(*args, **kwargs)
    # Preserve intermediate policies because success_rate alone saturates at
    # one early for a single motion and cannot distinguish smooth checkpoints.
    cfg.save_epoch_checkpoint_every = 50
    cfg.evaluator.quality_checkpoint_score = True
    cfg.evaluator.quality_score_success_weight = 10.0
    cfg.evaluator.quality_score_gt_weight = 1.0
    cfg.evaluator.quality_score_gr_weight = 0.25
    cfg.evaluator.quality_score_jerk_weight = 1.0e-4
    cfg.evaluator.quality_score_opening_jerk_weight = 2.0e-4
    cfg.evaluator.quality_score_action_delta_weight = 0.1
    cfg.evaluator.eval_action_ema_alpha = 0.6
    return cfg


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        foot_sliding_rew_factory,
        relative_body_ori_rew_factory,
        relative_body_pos_rew_factory,
    )

    cfg = robust_experiment.env_config(*args, **kwargs)
    # Preserve the standard total position/orientation weights while separating
    # world trajectory following from root-heading-relative articulation.
    cfg.reward_components["gt_rew"].static_params["weight"] = 0.30
    cfg.reward_components["gr_rew"].static_params["weight"] = 0.20
    cfg.reward_components["relative_body_pos_rew"] = relative_body_pos_rew_factory(
        weight=0.20,
        sigma=0.30,
        use_region_weights=False,
    )
    cfg.reward_components["relative_body_ori_rew"] = relative_body_ori_rew_factory(
        weight=0.10,
        sigma=0.40,
        use_region_weights=False,
    )
    cfg.reward_components["foot_sliding_rew"] = foot_sliding_rew_factory(
        weight=-0.05,
        min_value=-0.20,
        zero_during_grace_period=True,
    )
    return cfg
