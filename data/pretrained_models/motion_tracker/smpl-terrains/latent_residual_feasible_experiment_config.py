# SPDX-License-Identifier: Apache-2.0
"""Latent-residual tracker with terrain-feasibility-aware position reward."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LATENT_EXPERIMENT_PATH = (
    Path(__file__).resolve().parent / "latent_residual_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_latent_residual_base", _LATENT_EXPERIMENT_PATH
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
        reference_penetration_metric_factory,
        terrain_feasible_gt_rew_factory,
    )

    cfg = latent_experiment.env_config(*args, **kwargs)
    # Replace only global-position tracking.  Every other reward is identical
    # to the standard terrain tracker so this remains a clean ablation.
    cfg.reward_components["gt_rew"] = terrain_feasible_gt_rew_factory(
        weight=0.5,
        coefficient=-25.0,
        penetration_margin=0.02,
        penetration_scale=0.05,
        min_z_weight=0.05,
    )
    cfg.reward_components["reference_penetration_depth"] = (
        reference_penetration_metric_factory(fraction=False)
    )
    cfg.reward_components["reference_penetration_fraction"] = (
        reference_penetration_metric_factory(fraction=True)
    )
    return cfg
