# SPDX-License-Identifier: Apache-2.0
"""Noise-robust selective-plasticity adaptation for video motion tracking.

The official tracker remains the initialization and most of its trunk stays
frozen.  Three zero-initialized side adapters receive a sparse future command
window, while the final two hidden layers and action head are allowed to move
under a weak L2-SP anchor.  This closes the expressivity gap of fully frozen
adapters without paying the cost and instability of per-clip training from
scratch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "layerwise_adapter_contact_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_reliability_plasticity_base", _BASE_PATH
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

    # Keep the pretrained policy's one-step input exactly compatible, and add
    # a sparse 33/67/133/267 ms side context for anticipatory corrections.
    cfg.control_components["mimic"].future_steps = [1, 2, 4, 8]
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

    # Avoid solving tracking by suppressing motion amplitude.  Contact quality
    # remains constrained, but tracking velocities carry more of the objective.
    cfg.reward_components["action_smoothness"] = action_smoothness_factory(
        weight=-0.01,
        ignore_first_steps=1,
    )
    cfg.reward_components["foot_sliding_rew"] = foot_sliding_rew_factory(
        weight=-0.03,
        min_value=-0.25,
        zero_during_grace_period=True,
    )
    cfg.reward_components["pow_rew"].static_params["weight"] = -1.0e-5
    cfg.reward_components["gv_rew"].static_params["weight"] = 0.15
    cfg.reward_components["gav_rew"].static_params["weight"] = 0.10
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    actor = cfg.model.actor
    actor.adapter_layer_indices = [1, 3, 5]
    actor.bottleneck_size = 256
    actor.adapter_scale = 0.25
    actor.context_key = "mimic_future_context"
    actor.reliability_key = "reference_reliability"
    actor.context_bottleneck_size = 256
    actor.context_scale = 0.15
    actor.trainable_tail_layers = 2
    actor.pretrained_anchor_coef = 1.0e-3

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
    cfg.model.actor_optimizer.lr = 2.0e-5
    cfg.model.actor_optimizer.weight_decay = 1.0e-5
    cfg.model.critic_optimizer.lr = 1.0e-5
    cfg.adaptive_lr.enabled = True
    cfg.adaptive_lr.desired_kl = 0.01
    cfg.adaptive_lr.min_lr = 2.0e-6
    cfg.adaptive_lr.max_lr = 5.0e-5
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
    # Match the training dynamics.  The previous inference-only 0.02 armature
    # contributed visible lag on seated and contact-rich motions.
    for control_info in robot_cfg.control.control_info.values():
        control_info.armature = 0.01
