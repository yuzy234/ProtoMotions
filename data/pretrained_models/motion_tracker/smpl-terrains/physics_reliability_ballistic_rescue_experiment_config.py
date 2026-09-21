# SPDX-License-Identifier: Apache-2.0
"""Rescue adaptation for support-anchored 3-D ballistic video references.

This keeps the frozen large-dataset tracker, reliability-factorized root
tracking, layer adapters, and zero-initialized action residual from the rescue
experiment.  The only behavioral change is a wider *training-only* tracking
termination.  A cold policy misses the first landing by roughly 0.3 m and the
second take-off by more than 1 m; terminating at 1.25 m prevents full-sequence
rollouts from ever observing the wall-contact phase.  Overlapping clips still
cover every phase, while the evaluator keeps the strict 0.5 m success gate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "physics_reliability_rescue_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_ballistic_rescue_base", _BASE_PATH
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
    cfg.termination_components["tracking_error"].static_params[
        "error_threshold"
    ] = 2.0
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    # Report the controller itself: no deployment-time action filtering or
    # settling pre-roll is allowed to hide a bad take-off/reset state.
    cfg.evaluator.eval_action_ema_alpha = None
    cfg.evaluator.eval_action_ema_alpha_min = None
    cfg.evaluator.eval_action_ema_alpha_max = None
    cfg.evaluator.eval_pre_roll_steps = 0
    return cfg
