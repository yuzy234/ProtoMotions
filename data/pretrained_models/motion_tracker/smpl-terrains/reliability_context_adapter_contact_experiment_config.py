# SPDX-License-Identifier: Apache-2.0
"""Frozen tracker with a ten-frame reliability-gated context adapter.

Unlike a direct action residual, temporal corrections are injected into the
last two hidden representations and must pass through the pretrained action
head.  The whole pretrained policy remains frozen.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "layerwise_adapter_contact_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_reliability_context_adapter_base", _BASE_PATH
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
        mimic_target_poses_future_rel_factory,
        mimic_target_poses_max_coords_factory,
        reference_reliability_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.control_components["mimic"].future_steps = list(range(1, 11))
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
    # Keep the proven physical regularizers from the old adapter and add
    # explicit velocity fidelity to reject motion-amplitude collapse.
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
    actor.context_bottleneck_size = 256
    actor.context_scale = 0.25
    actor.action_context_bottleneck_size = 0
    actor.action_context_scale = 0.0
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
    cfg.model.actor_optimizer.lr = 2.0e-5
    cfg.model.actor_optimizer.weight_decay = 1.0e-5
    cfg.model.critic_optimizer.lr = 5.0e-6
    cfg.adaptive_lr.enabled = False
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
    for control_info in robot_cfg.control.control_info.values():
        control_info.armature = 0.01
