# SPDX-License-Identifier: Apache-2.0
"""Reliability-anchored online phase adaptation for noisy video motion."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_plasticity_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_adaptive_phase_base", _BASE_PATH
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
    mimic = cfg.control_components["mimic"]
    mimic.adaptive_phase_enabled = True
    mimic.adaptive_phase_rates = [1.0, 1.5, 2.0, 2.5, 3.0]
    mimic.adaptive_phase_unreliable_below = 0.25
    mimic.adaptive_phase_reliable_above = 0.75
    mimic.adaptive_phase_contact_lock_reliability = 0.80
    mimic.adaptive_phase_pose_scale = 0.20
    mimic.adaptive_phase_prior_weight = 0.25
    mimic.adaptive_phase_change_weight = 0.05
    # Training sees enough of failed rollouts to learn a correction; evaluator
    # success and checkpoint ranking remain the inherited strict 0.5 m metric.
    cfg.termination_components["tracking_error"].static_params[
        "error_threshold"
    ] = 1.25
    return cfg


def agent_config(*args, **kwargs):
    return base_experiment.agent_config(*args, **kwargs)
