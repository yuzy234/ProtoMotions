# SPDX-License-Identifier: Apache-2.0
"""Contact-refined adapter with targeted lower-body action acceleration control."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "layerwise_adapter_contact_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_layerwise_adapter_lower_body_accel_base", _BASE_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


# The SMPL action is ordered as three coordinates per non-root body.  Target
# knees, ankles, and toes, which dominate the measured angular jerk, while
# leaving hips, torso, and arms free to make larger tracking corrections.
LOWER_LEG_ACTION_INDICES = list(range(3, 12)) + list(range(15, 24))


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import action_acceleration_factory

    cfg = base_experiment.env_config(*args, **kwargs)
    if cfg.num_state_history_steps < 2:
        raise ValueError("Lower-body acceleration reward requires two history steps.")
    cfg.reward_components["lower_body_action_acceleration"] = (
        action_acceleration_factory(
            weight=-0.01,
            indices=LOWER_LEG_ACTION_INDICES,
            ignore_first_steps=2,
        )
    )
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    # This is a conservative refinement of an already useful adapter.
    cfg.model.actor_optimizer.lr = 2e-5
    cfg.model.critic_optimizer.lr = 2e-6
    return cfg
