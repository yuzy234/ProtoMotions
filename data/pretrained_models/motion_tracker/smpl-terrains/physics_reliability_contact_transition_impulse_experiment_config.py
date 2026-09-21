# SPDX-License-Identifier: Apache-2.0
"""Full-leg contact expert with a dense, timing-aware impulse objective.

The full-leg expert can physically produce upward velocity, but the generic
exponential body-velocity reward saturates once a difficult take-off is missed.
This ablation adds a robust current root-vz error only while reliable reference
support precedes take-off.  It does not modify the reference, apply forces, or
filter deployment actions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_contact_transition_fullleg_expert_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_contact_transition_impulse_base",
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
        contact_transition_vertical_velocity_error_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.reward_components["takeoff_vertical_velocity_error"] = (
        contact_transition_vertical_velocity_error_factory(
            weight=-0.25,
            support_body_ids=base_experiment.SUPPORT_BODY_IDS,
            vertical_speed_scale=3.0,
            horizon_weights=base_experiment.HORIZON_WEIGHTS,
            huber_delta=0.5,
            max_error=5.0,
        )
    )
    return cfg
