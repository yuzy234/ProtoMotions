# SPDX-License-Identifier: Apache-2.0
"""Learn hard contact transitions before demanding a full vault rollout.

The two-second curriculum used by the COM-ballistic tracker makes every
window fail on the screencast vault.  Once all windows have the same failure
rate, failure-weighted sampling no longer identifies the second take-off as
the useful hard example.  This ablation changes only episode organization:

* one-second overlapping windows isolate individual contact transitions;
* a 0.75 m training-only cutoff makes local failure statistics informative;
* twenty percent of environments still start at frame zero and run the full
  sequence, so the controller cannot solve only reference-state resets.

The actor, rewards, COM-ballistic reference, five PPO mini-epochs, and strict
0.5 m full-motion evaluator are inherited unchanged.  Deployment continues to
use raw policy actions without EMA or filtering.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_com_ballistic_high_update_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_com_ballistic_local_curriculum_base",
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
    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.motion_manager.clip_duration = 1.0
    cfg.motion_manager.clip_stride = 0.1
    cfg.motion_manager.failure_sampling_mix = 0.9
    cfg.motion_manager.max_failure_weight_ratio = 8.0
    cfg.motion_manager.full_motion_sampling_probability = 0.20
    cfg.termination_components["tracking_error"].static_params[
        "error_threshold"
    ] = 0.75
    return cfg
