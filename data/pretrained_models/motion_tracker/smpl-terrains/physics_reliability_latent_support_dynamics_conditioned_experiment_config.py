# SPDX-License-Identifier: Apache-2.0
"""Dynamics-conditioned tracking of a physically calibrated video reference.

The latent-support/reachability calibration makes the offline trajectory
physically feasible.  This variant additionally exposes confidence-weighted
future root vertical velocity to the temporal adapter.  Future poses and
height anchors alone are ambiguous at takeoff; desired vertical velocity tells
the policy early enough to prepare the support impulse.  It is an observation,
not action smoothing, force assistance, or a deployment-time trajectory edit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_latent_support_reachable_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_latent_support_dynamics_base",
    _BASE_PATH,
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
agent_config = base_experiment.agent_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        reliability_gated_mimic_target_poses_future_rel_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.observation_components["mimic_future_context"] = (
        reliability_gated_mimic_target_poses_future_rel_factory(
            minimum_global_weight=0.10,
            gravity_axis_only=True,
            include_reliability=True,
            include_trusted_vertical_anchors=True,
            include_trusted_vertical_velocity=True,
        )
    )
    return cfg
