# SPDX-License-Identifier: Apache-2.0
"""Long-horizon trusted-anchor planning across missing-contact gaps.

The short temporal adapter sees only 267 ms, but monocular occlusion can hide
support for more than a second.  This variant samples commands out to 1.6 s.
Unknown intermediate root heights remain gated; the side policy receives only
confidence-weighted height offsets of future reliable anchors, together with
the complete root-relative articulation.  It can therefore anticipate a
landing or hand support without imitating the non-physical path between them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_openmode_temporal_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_openmode_longhorizon_base", _BASE_PATH
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
    # At 30 Hz this spans 33 ms to 1.6 s while keeping the near future dense.
    cfg.control_components["mimic"].future_steps = [1, 2, 4, 8, 16, 24, 32, 48]
    cfg.observation_components["mimic_future_context"] = (
        reliability_gated_mimic_target_poses_future_rel_factory(
            minimum_global_weight=0.10,
            gravity_axis_only=True,
            include_reliability=True,
            include_trusted_vertical_anchors=True,
        )
    )
    return cfg
