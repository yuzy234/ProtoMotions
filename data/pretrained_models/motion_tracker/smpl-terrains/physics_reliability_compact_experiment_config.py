# SPDX-License-Identifier: Apache-2.0
"""Compact root-reliability adaptation for noisy video motions.

The root-gauge confidence and mixed short/full-clip curriculum are retained,
but policy plasticity is deliberately limited to the two bottleneck adapters
that were stable in the original scene tracker.  Reliability already enters
the target encoding and objective; a second large future-context branch is not
needed to make the policy uncertainty aware.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_plasticity_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_compact_base", _BASE_PATH
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
    # The compact actor does not consume the explicit multi-horizon side
    # channel.  Keep the standard one-step pretrained command and the scalar
    # reliability observation used by diagnostics.
    cfg.control_components["mimic"].future_steps = 1
    cfg.observation_components.pop("mimic_future_context", None)
    # First establish a stable executable controller. A targeted acceleration
    # refinement can be applied to the selected checkpoint afterwards; using
    # it from epoch zero slowed recovery from falls in the large-adapter run.
    cfg.reward_components.pop("lower_body_action_acceleration", None)
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    actor = cfg.model.actor
    actor.adapter_layer_indices = [3, 5]
    actor.adapter_scale = 0.15
    actor.context_key = None
    actor.context_scale = 0.0
    actor.reliability_key = None
    actor.uncertainty_gated_adapters = False
    actor.adapter_uncertainty_floor = 0.0

    unused = {"mimic_future_context", "reference_reliability"}
    actor.in_keys = [key for key in actor.in_keys if key not in unused]
    cfg.model.in_keys = [key for key in cfg.model.in_keys if key not in unused]
    return cfg
