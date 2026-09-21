# SPDX-License-Identifier: Apache-2.0
"""Two-depth bottleneck adaptation for noisy-reference motion tracking."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from protomotions.agents.mimic.layerwise_adapter_tracker_actor import (
    LayerwiseAdapterTrackerActorConfig,
)


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "latent_residual_reliability_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_layerwise_adapter_reliability_base", _BASE_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
env_config = base_experiment.env_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator


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
    """Deploy with CRISP's more inertial SMPL joints.

    Cold-rollout ablations show a Pareto improvement when the adapter is
    trained with the official 0.01 armature and deployed at 0.02. Training at
    0.02 from scratch instead increases body/root jerk, so this deliberately
    remains an inference-only dynamics refinement.
    """
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
        control_info.armature = 0.02


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    latent_actor = cfg.model.actor
    frozen_actor = latent_actor.frozen_actor
    checkpoint_path = latent_actor.frozen_actor_checkpoint

    cfg.model.actor = LayerwiseAdapterTrackerActorConfig(
        mu_key=frozen_actor.mu_key,
        in_keys=list(frozen_actor.in_keys),
        out_keys=list(dict.fromkeys(frozen_actor.out_keys + ["raw_mean_action"])),
        num_out=frozen_actor.num_out,
        actor_logstd=latent_actor.actor_logstd,
        learnable_std=True,
        frozen_actor=frozen_actor,
        frozen_actor_checkpoint=checkpoint_path,
        adapter_layer_indices=[3, 5],
        hidden_size=1024,
        bottleneck_size=256,
        adapter_scale=0.15,
        residual_noise_scale=0.2,
        min_logstd=-3.0,
        max_logstd=-0.5,
    )
    cfg.model.out_keys = list(dict.fromkeys(cfg.model.out_keys + ["raw_mean_action"]))
    cfg.model.actor_optimizer.lr = 5e-5
    cfg.model.actor_optimizer.weight_decay = 1e-5
    cfg.model.critic_optimizer.lr = 5e-6
    return cfg
