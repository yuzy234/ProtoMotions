# SPDX-License-Identifier: Apache-2.0
"""A deterministic contact-transition expert on top of the frozen prior.

The expert is identically zero outside reliable support-to-flight transitions
and can alter only lower-body joint targets. During those same short windows,
smoothness regularization is reduced but never removed. This lets PPO learn a
necessary support impulse while preserving the large-dataset motion prior on
ordinary frames and keeping deployment free of filters or force assistance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_contact_demand_exploration_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_contact_transition_expert_base",
    _BASE_PATH,
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


SUPPORT_BODY_IDS = [3, 4, 7, 8]
LOWER_BODY_ACTION_INDICES = list(range(3, 12)) + list(range(15, 24))
HORIZON_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        demand_relaxed_action_acceleration_factory,
        demand_relaxed_action_smoothness_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.reward_components["action_smoothness"] = (
        demand_relaxed_action_smoothness_factory(
            weight=-0.025,
            support_body_ids=SUPPORT_BODY_IDS,
            vertical_speed_scale=3.0,
            horizon_weights=HORIZON_WEIGHTS,
            minimum_multiplier=0.1,
            ignore_first_steps=1,
        )
    )
    cfg.reward_components["lower_body_action_acceleration"] = (
        demand_relaxed_action_acceleration_factory(
            weight=-0.005,
            indices=LOWER_BODY_ACTION_INDICES,
            support_body_ids=SUPPORT_BODY_IDS,
            vertical_speed_scale=3.0,
            horizon_weights=HORIZON_WEIGHTS,
            minimum_multiplier=0.1,
            ignore_first_steps=2,
        )
    )
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    actor = cfg.model.actor
    actor.demand_action_bottleneck_size = 256
    actor.demand_action_scale = 1.0
    actor.demand_action_indices = LOWER_BODY_ACTION_INDICES
    return cfg
