# SPDX-License-Identifier: Apache-2.0
"""Dynamic adaptation for mass-centre-corrected noisy video motion.

The generic walking/sitting regularizers used by the stable adapter can make a
vault policy minimize power and command variation instead of producing the
required take-off impulse.  This variant keeps those terms as weak priors,
raises velocity tracking, and re-enables a small foot-contact loss now that
labels come from the reconstructed SMPL surface rather than noisy 2-D contact
predictions.  Deployment still uses raw policy actions with no EMA/filter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "physics_reliability_ballistic_rescue_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_com_ballistic_base", _BASE_PATH
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
    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.reward_components["action_smoothness"].static_params["weight"] = -0.005
    cfg.reward_components["lower_body_action_acceleration"].static_params[
        "weight"
    ] = -0.001
    cfg.reward_components["foot_sliding_rew"].static_params["weight"] = -0.01
    cfg.reward_components["pow_rew"].static_params["weight"] = -1.0e-5
    cfg.reward_components["gv_rew"].static_params["weight"] = 0.25
    cfg.reward_components["contact_match_rew"].static_params["weight"] = -0.03
    return cfg


def agent_config(*args, **kwargs):
    return base_experiment.agent_config(*args, **kwargs)
