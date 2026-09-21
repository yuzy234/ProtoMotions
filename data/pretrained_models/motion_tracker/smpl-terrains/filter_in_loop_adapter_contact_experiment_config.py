# SPDX-License-Identifier: Apache-2.0
"""Parameter-efficient adaptation with deployment filtering in the RL loop.

The former pipeline trained on unfiltered actions and enabled a 0.6 EMA only
during evaluation. That makes the deployment plant slower than the plant seen
by PPO. Here the same causal filter is part of every environment step, allowing
the small layerwise adapter to compensate its lag without replacing the broad
motion prior of the pretrained tracker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().parent / "layerwise_adapter_contact_experiment_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_filter_in_loop_base", _BASE_PATH
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
agent_config = base_experiment.agent_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator


def env_config(*args, **kwargs):
    cfg = base_experiment.env_config(*args, **kwargs)
    cfg.action_lowpass_alpha = 0.6
    # Preserve the pretrained controller's original closed-loop state
    # distribution, then introduce deployment dynamics gradually.
    cfg.action_lowpass_anneal_epochs = 75
    return cfg


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    base_experiment.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    # Inference deploys the final filter directly; annealing is training-only.
    env_cfg.action_lowpass_anneal_epochs = 0
    # The environment already applies exactly one filter. Disable the legacy
    # evaluator-only copy to avoid filtering the command twice.
    agent_cfg.evaluator.eval_action_ema_alpha = None
    agent_cfg.evaluator.eval_action_ema_alpha_min = None
    agent_cfg.evaluator.eval_action_ema_alpha_max = None
    agent_cfg.evaluator.eval_action_ema_contact_alpha = None
