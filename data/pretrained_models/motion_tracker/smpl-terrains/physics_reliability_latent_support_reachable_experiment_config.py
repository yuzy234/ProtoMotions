# SPDX-License-Identifier: Apache-2.0
"""Long-horizon adaptation for a latent-support reachable reference.

This is intentionally policy-identical to the open-set long-horizon ablation.
The only experimental variable is the offline reference: seed-connected foot
support recovery, scene-gauge projection, reachability-aware phase calibration,
and whole-body-COM ballistic flight.  Keeping the actor, rewards, optimizer,
curriculum, and raw deployment actions fixed isolates whether the reference
repair removes the impossible hovering target.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_openmode_longhorizon_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_latent_support_reachable_base",
    _BASE_PATH,
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
env_config = base_experiment.env_config
agent_config = base_experiment.agent_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides
