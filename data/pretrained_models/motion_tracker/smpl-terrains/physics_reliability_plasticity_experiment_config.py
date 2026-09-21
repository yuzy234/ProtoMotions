# SPDX-License-Identifier: Apache-2.0
"""Physics-reliability-gated adaptation for noisy video root trajectories.

This experiment combines a general pretrained tracker with selective policy
plasticity.  The MotionLib supplies support/flight consistency confidence for
the global root path.  High-confidence frames retain absolute tracking; low-
confidence frames smoothly fall back to root-relative articulation tracking
and let contact dynamics determine the world-space trajectory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "reliability_plasticity_contact_experiment_config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "smpl_terrains_physics_reliability_plasticity_base", _BASE_PATH
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
    from protomotions.envs.component_factories import (
        action_acceleration_factory,
        action_smoothness_factory,
        physical_reference_reliability_factory,
        reliability_gated_mimic_target_poses_max_coords_factory,
        reliability_blended_position_rew_factory,
        reliability_blended_tracking_error_term_factory,
        reliability_blended_velocity_rew_factory,
    )

    cfg = base_experiment.env_config(*args, **kwargs)
    # Confidence is calibrated into a control weight.  Frames at or above 0.75
    # use the exact pretrained absolute command; only weak/ambiguous evidence is
    # relaxed.  This avoids accumulating root drift on clean MPH11/N0Sofa spans.
    mimic_control = cfg.control_components["mimic"]
    mimic_control.reference_reliability_gate_low = 0.35
    mimic_control.reference_reliability_gate_high = 0.75
    mimic_control.reference_reliability_gate_smoothstep = True
    # A quarter of environments run the complete sequence.  Short windows
    # retain phase coverage, while full episodes expose accumulated root drift
    # to PPO instead of hiding it behind a reset every two seconds.
    cfg.motion_manager.full_motion_sampling_probability = 0.25
    cfg.observation_components["reference_reliability"] = (
        physical_reference_reliability_factory()
    )
    cfg.observation_components["mimic_target_poses"] = (
        reliability_gated_mimic_target_poses_max_coords_factory(
            with_velocities=True,
            future_steps=1,
            minimum_global_weight=0.10,
            gravity_axis_only=True,
        )
    )
    cfg.reward_components["gt_rew"] = reliability_blended_position_rew_factory(
        weight=0.50,
        global_coefficient=-25.0,
        relative_coefficient=-25.0,
        minimum_absolute_weight=0.10,
        gravity_axis_only=True,
    )
    # Root height is already represented by the reliability-blended position
    # term.  A second unconditional height reward would silently restore the
    # noisy absolute target exactly where confidence says not to trust it.
    cfg.reward_components["rh_rew"].static_params["weight"] = 0.0
    cfg.reward_components["gv_rew"] = reliability_blended_velocity_rew_factory(
        weight=0.15,
        global_coefficient=-0.5,
        relative_coefficient=-0.5,
        minimum_absolute_weight=0.10,
        gravity_axis_only=True,
    )
    # These are policy-training objectives, not a deployment-time action
    # filter.  The first difference discourages large alternating commands,
    # while the targeted second difference addresses the knees/ankles/toes
    # that dominate measured body jerk without damping hips, torso, or arms.
    cfg.reward_components["action_smoothness"] = action_smoothness_factory(
        weight=-0.025,
        ignore_first_steps=1,
    )
    lower_leg_action_indices = list(range(3, 12)) + list(range(15, 24))
    cfg.reward_components["lower_body_action_acceleration"] = (
        action_acceleration_factory(
            weight=-0.005,
            indices=lower_leg_action_indices,
            ignore_first_steps=2,
        )
    )
    # The previous weak regularizers allowed PPO to improve stochastic return
    # by issuing increasingly aggressive corrections.  Restore the measured
    # stable setting while leaving reference-contact matching disabled.
    cfg.reward_components["foot_sliding_rew"].static_params["weight"] = -0.08
    cfg.reward_components["pow_rew"].static_params["weight"] = -5.0e-5
    cfg.reward_components["contact_force_change_rew"].static_params["weight"] = 0.0
    cfg.termination_components["tracking_error"] = (
        reliability_blended_tracking_error_term_factory(
            threshold=0.50,
            minimum_absolute_weight=0.10,
            gravity_axis_only=True,
        )
    )
    return cfg


def agent_config(*args, **kwargs):
    from protomotions.envs.component_factories import (
        gt_error_factory,
        reliability_blended_gt_error_factory,
    )

    cfg = base_experiment.agent_config(*args, **kwargs)
    # Preserve the large-dataset tracker as an exact immutable prior.  Allowing
    # the last two pretrained layers to update looked harmless on random 2 s
    # clips but accumulated enough policy drift to make a 5 s rollout fall.
    # Zero-initialized adapters retain scene-specific plasticity without moving
    # any original actor parameter.
    cfg.model.actor.trainable_tail_layers = 0
    cfg.model.actor.pretrained_anchor_coef = 0.0
    # Root confidence decides how strongly the *world-space gauge* is tracked;
    # it must not suppress the generic scene/dynamics adapter.  The latter is
    # needed even on highly reliable frames.  Gating the entire adapter to 5%
    # produced a characteristic failure: short windows improved while the
    # deterministic full rollout accumulated drift and eventually fell.
    cfg.model.actor.uncertainty_gated_adapters = False
    cfg.model.actor.adapter_uncertainty_floor = 0.0
    # Keep actor and critic learning rates separate. ProtoMotions' shared KL
    # scheduler raises both optimizers to one common ceiling; in this setup it
    # silently changed the critic from 5e-6 to 3e-5 and correlated with the
    # late deterministic-rollout collapses seen in the adaptive-LR ablation.
    cfg.model.actor_optimizer.lr = 3.0e-5
    cfg.model.critic_optimizer.lr = 5.0e-6
    cfg.adaptive_lr.enabled = False
    # Evaluation must expose the learned controller itself.  No action EMA or
    # deployment-time low-pass filter is used to manufacture smoothness.
    cfg.evaluator.eval_action_ema_alpha = None
    cfg.evaluator.eval_action_ema_alpha_min = None
    cfg.evaluator.eval_action_ema_alpha_max = None
    cfg.evaluator.quality_checkpoint_score = True
    # Keep raw global error as a diagnostic, but judge success/checkpoint
    # quality with the same uncertainty semantics used by training.  This lets
    # physics reject an unreliable shared root translation without rewarding
    # loss of articulation.
    cfg.evaluator.evaluation_components["gt_error"] = gt_error_factory()
    cfg.evaluator.evaluation_components["reliability_gt_error"] = (
        reliability_blended_gt_error_factory(
            threshold=0.50,
            minimum_absolute_weight=0.10,
            gravity_axis_only=True,
        )
    )
    cfg.evaluator.quality_score_position_metric = "reliability_gt_error"
    cfg.evaluator.quality_score_success_weight = 10.0
    cfg.evaluator.quality_score_gt_weight = 1.0
    cfg.evaluator.quality_score_gr_weight = 0.25
    cfg.evaluator.quality_score_jerk_weight = 2.0e-4
    cfg.evaluator.quality_score_opening_jerk_weight = 3.0e-4
    cfg.evaluator.quality_score_action_delta_weight = 0.15
    return cfg
