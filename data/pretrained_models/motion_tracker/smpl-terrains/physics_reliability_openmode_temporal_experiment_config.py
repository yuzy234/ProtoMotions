# SPDX-License-Identifier: Apache-2.0
"""Open-set root reliability with horizon-wise temporal aggregation.

Monocular contact is incomplete: ``not observed in contact`` must not imply
``free flight``.  The packaged motion therefore labels unsupported,
non-ballistic spans as an unknown physical mode and lowers their absolute
vertical confidence.  This experiment carries that confidence through every
future command horizon, so the temporal residual cannot reintroduce a noisy
root jump that the one-step reliability gate already rejected.

XY trajectory and root-relative articulation remain exact.  No action EMA,
low-pass filter, or deployment assistance is used.  One-second overlapping
clips retain contact-transition coverage, while complete rollouts preserve the
true prefix-state distribution.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_com_ballistic_local_curriculum_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_openmode_temporal_base", _BASE_PATH
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
    from protomotions.envs.component_factories import (
        reliability_gated_mimic_target_poses_future_rel_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.observation_components["mimic_future_context"] = (
        reliability_gated_mimic_target_poses_future_rel_factory(
            minimum_global_weight=0.10,
            gravity_axis_only=True,
            include_reliability=True,
        )
    )
    # Retain enough complete executions to expose drift without overwhelming
    # the one-second transition curriculum during initial adaptation.
    cfg.motion_manager.full_motion_sampling_probability = 0.25
    return cfg
