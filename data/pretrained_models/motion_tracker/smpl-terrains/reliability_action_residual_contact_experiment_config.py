# SPDX-License-Identifier: Apache-2.0
"""Reliability-conditioned temporal action residual for noisy video motion.

The general motion tracker is kept frozen.  Layerwise adapters retain the
small feature corrections used by the previous method, while a zero-initialized
action residual head receives the final policy feature and a sparse 0.33 s
future command window.  This adds clip-specific capacity without overwriting
the pretrained controller.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "layerwise_adapter_contact_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_reliability_action_residual_base", _BASE_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        foot_sliding_rew_factory,
        mimic_target_poses_future_rel_factory,
        mimic_target_poses_max_coords_factory,
        reference_reliability_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.control_components["mimic"].future_steps = [1, 2, 3, 4, 6, 8, 10]
    cfg.observation_components["mimic_target_poses"] = (
        mimic_target_poses_max_coords_factory(
            with_velocities=True,
            future_steps=1,
        )
    )
    cfg.observation_components["mimic_future_context"] = (
        mimic_target_poses_future_rel_factory()
    )
    cfg.observation_components["reference_reliability"] = (
        reference_reliability_factory(
            linear_speed_soft=4.0,
            angular_speed_soft=6.0,
            linear_change_scale=2.0,
            angular_change_scale=4.0,
            min_reliability=0.1,
        )
    )

    # Retain the successful physical regularization of the frozen-adapter
    # baseline, but strengthen velocity tracking so smoothness cannot be won by
    # reducing motion amplitude.
    cfg.reward_components["action_smoothness"] = action_smoothness_factory(
        weight=-0.025,
        ignore_first_steps=1,
    )
    cfg.reward_components["foot_sliding_rew"] = foot_sliding_rew_factory(
        weight=-0.06,
        min_value=-0.25,
        zero_during_grace_period=True,
    )
    cfg.reward_components["pow_rew"].static_params["weight"] = -3.0e-5
    cfg.reward_components["gv_rew"].static_params["weight"] = 0.15
    cfg.reward_components["gav_rew"].static_params["weight"] = 0.10
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    actor = cfg.model.actor
    actor.adapter_layer_indices = [3, 5]
    actor.bottleneck_size = 256
    actor.adapter_scale = 0.15
    actor.context_key = "mimic_future_context"
    actor.reliability_key = "reference_reliability"
    # Disable the weak hidden context paths in favor of a direct residual head.
    actor.context_scale = 0.0
    actor.action_context_bottleneck_size = 512
    actor.action_context_scale = 0.25
    actor.action_context_reliability_floor = 0.25
    actor.trainable_tail_layers = 0
    actor.pretrained_anchor_coef = 0.0

    actor.in_keys = list(
        dict.fromkeys(
            list(actor.in_keys)
            + ["mimic_future_context", "reference_reliability"]
        )
    )
    cfg.model.in_keys = list(
        dict.fromkeys(
            list(cfg.model.in_keys)
            + ["mimic_future_context", "reference_reliability"]
        )
    )
    cfg.model.actor_optimizer.lr = 5.0e-5
    cfg.model.actor_optimizer.weight_decay = 1.0e-5
    cfg.model.critic_optimizer.lr = 5.0e-6
    cfg.adaptive_lr.enabled = True
    cfg.adaptive_lr.desired_kl = 0.01
    cfg.adaptive_lr.min_lr = 5.0e-6
    cfg.adaptive_lr.max_lr = 1.0e-4
    return cfg


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    base_experiment.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    # Match training dynamics; inference-only extra armature produced lag in
    # seated/contact-rich motions.
    for control_info in robot_cfg.control.control_info.values():
        control_info.armature = 0.01
