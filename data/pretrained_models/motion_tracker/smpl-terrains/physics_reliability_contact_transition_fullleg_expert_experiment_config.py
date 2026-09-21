# SPDX-License-Identifier: Apache-2.0
"""Contact-transition expert with complete bilateral leg authority.

The preceding ablation accidentally reused the action indices of the
lower-*leg* acceleration regularizer.  Those indices omit both three-DoF hip
joints, even though hip extension is a primary contributor to take-off
impulse.  This experiment keeps every other variable fixed and gives the
demand-gated deterministic expert (and its locally relaxed acceleration
prior) authority over both complete legs: hips, knees, ankles, and toes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_contact_transition_expert_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_contact_transition_fullleg_base",
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


SUPPORT_BODY_IDS = base_experiment.SUPPORT_BODY_IDS
# SMPL robot DoF ordering starts with the complete left and right leg:
# L_Hip[0:3] ... L_Toe[9:12], R_Hip[12:15] ... R_Toe[21:24].
FULL_LEG_ACTION_INDICES = list(range(24))
HORIZON_WEIGHTS = base_experiment.HORIZON_WEIGHTS


def env_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        demand_relaxed_action_acceleration_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.reward_components["lower_body_action_acceleration"] = (
        demand_relaxed_action_acceleration_factory(
            weight=-0.005,
            indices=FULL_LEG_ACTION_INDICES,
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
    actor.demand_action_indices = FULL_LEG_ACTION_INDICES
    # Exploration and the deterministic expert must cover the same actuator
    # subspace so PPO can discover and retain hip-driven support impulses.
    actor.exploration_demand_action_indices = FULL_LEG_ACTION_INDICES
    return cfg
