# SPDX-License-Identifier: Apache-2.0
"""Layerwise tracker adaptation with reset-aware and measured-contact priors."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "layerwise_adapter_reliability_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_layerwise_adapter_contact_base", _BASE_PATH
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
        action_smoothness_factory,
        contact_force_change_rew_factory,
        foot_sliding_rew_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.reward_components["action_smoothness"] = action_smoothness_factory(
        weight=-0.025,
        ignore_first_steps=1,
    )
    # Estimated reference contacts are unreliable for this video motion. Use
    # measured simulator contact only for regularization.
    cfg.reward_components["contact_match_rew"].static_params["weight"] = 0.0
    cfg.reward_components["foot_sliding_rew"] = foot_sliding_rew_factory(
        weight=-0.08,
        min_value=-0.25,
        zero_during_grace_period=True,
    )
    cfg.reward_components["contact_force_change_rew"] = (
        contact_force_change_rew_factory(
            weight=-1.0e-5,
            min_value=-0.10,
            threshold=30.0,
            zero_during_grace_period=True,
        )
    )
    # A modest increase discourages high-frequency corrective torques without
    # imposing CRISP's full task-specific power coefficient in one jump.
    cfg.reward_components["pow_rew"].static_params["weight"] = -5.0e-5
    return cfg
