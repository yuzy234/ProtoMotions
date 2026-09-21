# SPDX-License-Identifier: Apache-2.0
"""Latent residual with feasible-global and root-relative pose tracking."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_FEASIBLE_PATH = (
    Path(__file__).resolve().parent / "latent_residual_feasible_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_latent_residual_feasible_base", _FEASIBLE_PATH
)
feasible_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(feasible_experiment)


terrain_config = feasible_experiment.terrain_config
scene_lib_config = feasible_experiment.scene_lib_config
motion_lib_config = feasible_experiment.motion_lib_config
agent_config = feasible_experiment.agent_config
configure_robot_and_simulator = feasible_experiment.configure_robot_and_simulator
apply_inference_overrides = feasible_experiment.apply_inference_overrides


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        relative_body_ori_rew_factory,
        relative_body_pos_rew_factory,
    )

    cfg = feasible_experiment.env_config(*args, **kwargs)

    # Keep the total position/orientation weights equal to the standard tracker,
    # but split each between world-space and root-heading-relative tracking.
    # This preserves trajectory following while making root estimation noise less
    # likely to force every limb to chase the same global error.
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
    return cfg
