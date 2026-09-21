# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Train a full-body action residual around the frozen SMPL terrain tracker.

This experiment reuses the standard ``motion_tracker/smpl-terrains`` environment,
reward, terrain, and critic setup.  The only architectural change is the actor:

    observations -> frozen tracker mean + residual MLP -> 69D PD target action

Do not pass the pretrained tracker via ``--checkpoint`` when using this experiment.
The frozen tracker checkpoint is loaded by ``ActionResidualTrackerActor``.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
from protomotions.agents.mimic.action_residual_tracker_actor import (
    ActionResidualTrackerActorConfig,
)
from protomotions.agents.mimic.root_residual_ppo_model import (
    PretrainedCriticPPOModelConfig,
)

_BASE_EXPERIMENT_PATH = Path(__file__).resolve().parent / "experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_base_experiment", _BASE_EXPERIMENT_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
env_config = base_experiment.env_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


def _root_residual_env_config(*args, **kwargs):
    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.preserve_reference_world_position = True
    # Match CRISP's terrain-aware reference initialization: do not hand PhysX
    # a reference pose whose feet are already inside the mesh.  This affects
    # only reset state, never the world-space reference motion/target itself.
    cfg.safe_reference_reset = True
    cfg.safe_reference_reset_margin = 0.02
    # The clip motion manager owns the 2-second training boundary. Keep the
    # normal episode limit long so full-motion inference is not reset early.
    from protomotions.envs.motion_manager.config import OverlappingClipMotionManagerConfig
    cfg.motion_manager = OverlappingClipMotionManagerConfig(
        clip_motion_id=0,
        clip_duration=2.0,
        clip_stride=1.0,
        failure_sampling_mix=0.8,
        init_start_prob=0.0,
        resample_on_reset=True,
    )
    return cfg


env_config = _root_residual_env_config


def agent_config(robot_config, env_config, args):
    cfg = base_experiment.agent_config(robot_config, env_config, args)

    frozen_actor = cfg.model.actor
    base_model_cfg = cfg.model
    obs_keys = list(frozen_actor.in_keys)

    checkpoint_path = (
        Path(__file__).resolve().parent / "last.ckpt"
    )

    cfg.model = PretrainedCriticPPOModelConfig(
        in_keys=base_model_cfg.in_keys,
        out_keys=base_model_cfg.out_keys,
        actor=base_model_cfg.actor,
        critic=base_model_cfg.critic,
        actor_optimizer=base_model_cfg.actor_optimizer,
        critic_optimizer=base_model_cfg.critic_optimizer,
        critic_checkpoint=str(checkpoint_path),
    )

    residual_model = MLPWithConcatConfig(
        in_keys=obs_keys + ["base_mu_raw"],
        out_keys=["residual_action_raw"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=frozen_actor.num_out,
        layers=[
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ],
    )

    cfg.model.actor = ActionResidualTrackerActorConfig(
        mu_key=frozen_actor.mu_key,
        in_keys=obs_keys,
        out_keys=frozen_actor.out_keys + ["raw_mean_action"],
        num_out=frozen_actor.num_out,
        # exp(-1.29056) * 0.2 == exp(-2.9), matching the pretrained
        # tracker's initial raw-action exploration exactly when residual=0.
        actor_logstd=-1.2905620875658997,
        learnable_std=True,
        frozen_actor=frozen_actor,
        residual_model=residual_model,
        frozen_actor_checkpoint=str(checkpoint_path),
        base_mu_key="base_mu_raw",
        residual_key="residual_action_raw",
        raw_mean_key="raw_mean_action",
        residual_mean_scale=1.0,
        residual_noise_scale=0.2,
        min_logstd=-3.0,
        max_logstd=-0.5,
    )
    cfg.model.out_keys = list(dict.fromkeys(cfg.model.out_keys + ["raw_mean_action"]))

    # The frozen tracker parameters have requires_grad=False; only the residual
    # MLP and residual logstd are optimized.
    cfg.model.actor_optimizer.lr = 5e-5
    cfg.model.critic_optimizer.lr = 1e-5
    return cfg
