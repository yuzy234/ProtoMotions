# SPDX-License-Identifier: Apache-2.0
"""Data-efficient PPO adaptation for COM-ballistic video references.

The default parameter-efficient run intentionally uses one PPO mini-epoch and
a nearly frozen critic.  That is stable on walking and sitting clips, but it
under-uses every expensive simulator batch on an out-of-distribution vault.
This ablation changes only optimizer-side sample reuse: the reference, reward,
frozen pretrained actor, adapters, direct action residual, and raw deployment
actions are identical to the COM-ballistic experiment.

Five mini-epochs match the update ratio used by ReActor while retaining PPO's
0.2 clipping.  The critic uses ProtoMotions' official terrain-tracker learning
rate so advantages can follow the new reliability-shaped reward.  The actor
rate is slightly reduced because each batch is now visited five times.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_com_ballistic_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_com_ballistic_high_update_base",
    _BASE_PATH,
)
base_experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base_experiment)


terrain_config = base_experiment.terrain_config
scene_lib_config = base_experiment.scene_lib_config
motion_lib_config = base_experiment.motion_lib_config
env_config = base_experiment.env_config
configure_robot_and_simulator = base_experiment.configure_robot_and_simulator
apply_inference_overrides = base_experiment.apply_inference_overrides


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    cfg.num_mini_epochs = 5
    cfg.model.actor_optimizer.lr = 3.0e-5
    cfg.model.critic_optimizer.lr = 1.0e-4
    cfg.entropy_coef = 2.5e-3
    # Multiple passes make very large per-pass gradients unnecessary.  This
    # protects the pretrained residual initialization without action filtering.
    cfg.gradient_clip_val = 5.0
    return cfg
