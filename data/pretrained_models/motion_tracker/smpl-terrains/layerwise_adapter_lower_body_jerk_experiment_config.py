# SPDX-License-Identifier: Apache-2.0
"""Contact-refined adapter with direct lower-body angular-jerk control."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "layerwise_adapter_contact_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_layerwise_adapter_lower_body_jerk_base", _BASE_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


# Knees, ankles, and toes are the six bodies with the largest residual local
# angular jerk. Their direct parents define the corresponding relative joints.
LOWER_LEG_BODY_INDICES = [2, 3, 4, 6, 7, 8]
LOWER_LEG_PARENT_INDICES = [1, 2, 3, 5, 6, 7]


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        relative_body_angular_jerk_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    if cfg.num_state_history_steps < 2:
        raise ValueError("Lower-body jerk reward requires two history steps.")
    cfg.reward_components["lower_body_relative_angular_jerk"] = (
        relative_body_angular_jerk_factory(
            weight=-0.02,
            body_indices=LOWER_LEG_BODY_INDICES,
            parent_indices=LOWER_LEG_PARENT_INDICES,
            ignore_first_steps=2,
            min_value=-0.15,
        )
    )
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    cfg.model.actor_optimizer.lr = 2e-5
    cfg.model.critic_optimizer.lr = 2e-6
    return cfg
