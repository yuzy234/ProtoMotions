# SPDX-License-Identifier: Apache-2.0
"""Ablation: hard-ramp the robust latent residual after every reset."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ROBUST_PATH = (
    Path(__file__).resolve().parent / "latent_residual_robust_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_latent_residual_robust_warmup_base", _ROBUST_PATH
)
robust_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(robust_experiment)


terrain_config = robust_experiment.terrain_config
scene_lib_config = robust_experiment.scene_lib_config
motion_lib_config = robust_experiment.motion_lib_config
configure_robot_and_simulator = robust_experiment.configure_robot_and_simulator
apply_inference_overrides = robust_experiment.apply_inference_overrides


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        reset_transition_progress_obs_factory,
    )

    cfg = robust_experiment.env_config(*args, **kwargs)
    cfg.observation_components["residual_warmup"] = (
        reset_transition_progress_obs_factory()
    )
    return cfg


def agent_config(robot_config, env_config, args):
    cfg = robust_experiment.agent_config(robot_config, env_config, args)
    warmup_key = "residual_warmup"
    cfg.model.actor.residual_model.in_keys = list(
        dict.fromkeys(cfg.model.actor.residual_model.in_keys + [warmup_key])
    )
    cfg.model.actor.residual_warmup_key = warmup_key
    cfg.model.actor.in_keys = list(
        dict.fromkeys(cfg.model.actor.in_keys + [warmup_key])
    )
    cfg.model.in_keys = list(dict.fromkeys(cfg.model.in_keys + [warmup_key]))
    return cfg
