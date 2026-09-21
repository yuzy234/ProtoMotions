# SPDX-License-Identifier: Apache-2.0
"""Latent residual with a terrain-projected feasible global target."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LATENT_EXPERIMENT_PATH = (
    Path(__file__).resolve().parent / "latent_residual_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_latent_residual_projected_base", _LATENT_EXPERIMENT_PATH
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
        reference_feasible_lift_metric_factory,
        reference_penetration_metric_factory,
        terrain_projected_gt_rew_factory,
        terrain_projected_rh_rew_factory,
    )

    cfg = latent_experiment.env_config(*args, **kwargs)
    clearance_margin = 0.02
    max_lift = 0.15

    # Match safe_reference_reset: use one minimum whole-body Z lift for both
    # global position and root-height targets.  Other tracker rewards stay fixed.
    cfg.reward_components["gt_rew"] = terrain_projected_gt_rew_factory(
        weight=0.5,
        coefficient=-25.0,
        clearance_margin=clearance_margin,
        max_lift=max_lift,
    )
    cfg.reward_components["rh_rew"] = terrain_projected_rh_rew_factory(
        weight=0.1,
        coefficient=-20.0,
        clearance_margin=clearance_margin,
        max_lift=max_lift,
    )
    cfg.reward_components["reference_feasible_lift"] = (
        reference_feasible_lift_metric_factory(
            clearance_margin=clearance_margin,
            max_lift=max_lift,
        )
    )
    cfg.reward_components["reference_penetration_depth"] = (
        reference_penetration_metric_factory(fraction=False)
    )
    cfg.reward_components["reference_penetration_fraction"] = (
        reference_penetration_metric_factory(fraction=True)
    )
    return cfg
