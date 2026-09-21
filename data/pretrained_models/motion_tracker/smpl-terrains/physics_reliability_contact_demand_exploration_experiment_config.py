# SPDX-License-Identifier: Apache-2.0
"""Contact-transition-directed exploration for noisy video motion tracking.

The deterministic actor and deployment interface are unchanged.  During PPO
sampling only, lower-body exploration is increased when the reference feet are
still supported and a reliable near-future command requires a sharp upward
velocity increase.  This targets the preparation impulse that a frozen motion
prior otherwise never discovers, without adding global noise, action filtering,
EMA, or inference-time force assistance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_latent_support_dynamics_conditioned_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_contact_demand_base", _BASE_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        contact_conditioned_takeoff_demand_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    # MimicControl exposes [1, 2, 4, 8, 16, 24, 32, 48] frame horizons.
    # Keep exploration local to the approximately 0.5 s preparation window;
    # horizons beyond it remain useful to the deterministic context adapter.
    cfg.observation_components["takeoff_exploration_demand"] = (
        contact_conditioned_takeoff_demand_factory(
            support_body_ids=[3, 4, 7, 8],
            vertical_speed_scale=3.0,
            horizon_weights=[1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
        )
    )
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    actor = cfg.model.actor
    demand_key = "takeoff_exploration_demand"
    actor.exploration_demand_key = demand_key
    actor.exploration_demand_max_multiplier = 4.0
    actor.exploration_demand_action_indices = (
        list(range(3, 12)) + list(range(15, 24))
    )
    # PPOModel validates the union of actor/critic inputs against this top-level
    # contract before the first TensorDict is materialized.
    cfg.model.in_keys = list(dict.fromkeys(list(cfg.model.in_keys) + [demand_key]))
    return cfg
