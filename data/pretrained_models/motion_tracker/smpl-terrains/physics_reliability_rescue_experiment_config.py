# SPDX-License-Identifier: Apache-2.0
"""High-plasticity rescue path for physically difficult video motions.

The regular physics-reliability experiment keeps the pretrained tracker fully
frozen and adapts it through hidden bottlenecks.  That is deliberately stable
for walking and sitting, but can be too weak for an out-of-distribution vault:
the hidden corrections still have to pass through the frozen action head.

This variant retains the immutable pretrained policy and its layer adapters,
then adds a zero-initialized action residual conditioned on the policy state
and sparse future articulation context.  A wider tracking-error termination
only keeps failed training rollouts alive long enough to expose later phases;
checkpoint success and ranking still use the strict 0.5 m evaluator metric.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "physics_reliability_plasticity_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_rescue_base", _BASE_PATH
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
    # The strict 0.5 m cutoff resets a vault after roughly one third of the
    # clip, so PPO never observes its wall-contact and landing phases.  This
    # threshold controls rollout truncation only; evaluator success remains at
    # 0.5 m through reliability_gt_error.
    cfg.termination_components["tracking_error"].static_params[
        "error_threshold"
    ] = 1.25
    return cfg


def agent_config(*args, **kwargs):
    cfg = base_experiment.agent_config(*args, **kwargs)
    actor = cfg.model.actor
    actor.action_context_bottleneck_size = 512
    actor.action_context_scale = 0.35
    # Reference height has already been reliability-gated in the target
    # encoder.  Do not suppress this dynamics-rescue path a second time on the
    # uncertain frames where it is most needed.
    actor.action_context_reliability_floor = 1.0
    cfg.model.actor_optimizer.lr = 5.0e-5
    cfg.model.actor_optimizer.weight_decay = 1.0e-5
    cfg.model.critic_optimizer.lr = 5.0e-6
    cfg.adaptive_lr.enabled = False
    return cfg
