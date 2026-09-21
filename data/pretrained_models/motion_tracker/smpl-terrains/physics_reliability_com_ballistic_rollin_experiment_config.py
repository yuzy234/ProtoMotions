# SPDX-License-Identifier: Apache-2.0
"""Roll-in consolidation after learning local contact transitions.

The one-second curriculum makes the difficult vault windows learnable when
they are initialized from the reference state, but those skills do not yet
transfer to the state distribution produced by executing the complete prefix.
This continuation keeps the same strict local windows and all optimization
settings while assigning 65% of environments to start-to-finish roll-ins.

It is intended as a warm start from the best local-curriculum checkpoint.  No
deployment assistance, action EMA, filtering, reward, or reference change is
introduced, so the experiment isolates state-distribution consolidation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_com_ballistic_local_curriculum_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_com_ballistic_rollin_base",
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
    cfg.motion_manager.full_motion_sampling_probability = 0.65
    return cfg
