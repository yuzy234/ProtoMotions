# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Configuration classes for evaluators."""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

from protomotions.envs.mdp_component import MdpComponent


@dataclass
class EvaluatorConfig:
    """Configuration for base evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    evaluation_components: Dict[str, MdpComponent] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of MdpComponent evaluation metrics for success/failure tracking."}
    )
    max_eval_steps: int = field(
        default=600,
        metadata={"help": "Maximum steps per evaluation episode.", "min": 1}
    )
    eval_metrics_every: Optional[int] = field(
        default=200,
        metadata={"help": "Evaluate metrics every N epochs. None = disabled.", "min": 1}
    )
    reset_training_envs_after_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "Reset all training environments at the next rollout boundary "
                "after an in-process evaluation. Physics contact warm-start "
                "state is backend-internal and cannot be snapshotted exactly."
            )
        },
    )


@dataclass
class MotionWeightsRulesConfig:
    """Configuration for motion weights update rule."""

    motion_weights_update_success_discount: float = field(
        default=0.999,
        metadata={"help": "Discount factor for successful motion weights.", "min": 0.0, "max": 1.0}
    )
    motion_weights_update_failure_discount: float = field(
        default=0.999,
        metadata={"help": "Discount for failed motions. 0 = set weight straight to 1.", "min": 0.0, "max": 1.0}
    )
    min_motion_weight: Union[float, str] = field(
        default="1/num_motions",
        metadata={"help": "Minimum weight for any motion. '1/num_motions' or float value."}
    )


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    save_predicted_motion_lib_every: Optional[int] = field(
        default=3,
        metadata={"help": "Save pred_motion_lib every M evals. None = disabled.", "min": 1}
    )
    motion_weights_rules: MotionWeightsRulesConfig = field(
        default_factory=MotionWeightsRulesConfig,
        metadata={"help": "Rules for updating motion sampling weights."}
    )
    park_inactive_envs: bool = field(
        default=True,
        metadata={
            "help": (
                "Move environments not used by a fixed-motion evaluation out "
                "of the collision scene. Disable only for backend diagnostics."
            )
        },
    )
    eval_action_ema_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "EMA smoothing factor for actions during evaluation only. "
                "Simulates deployment low-pass filtering. "
                "a_applied = alpha * a_policy + (1-alpha) * a_prev. "
                "None = disabled (raw actions). Typical values: 0.5-0.8."
                "Smaller alpha = more smoothing."
            ),
            "min": 0.0,
            "max": 1.0,
        }
    )
    eval_action_ema_alpha_min: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum adaptive EMA alpha used for abrupt action changes.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    eval_action_ema_alpha_max: Optional[float] = field(
        default=None,
        metadata={
            "help": "Maximum adaptive EMA alpha used for small action changes.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    eval_action_ema_delta_scale: float = field(
        default=0.03,
        metadata={
            "help": "Action RMS change scale controlling adaptive EMA decay.",
            "min": 1.0e-6,
        },
    )
    eval_action_ema_contact_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Maximum EMA alpha for a leg while its foot is in measured contact. "
                "None disables contact-aware leg filtering."
            ),
            "min": 0.0,
            "max": 1.0,
        },
    )
    eval_action_ema_contact_force_threshold: float = field(
        default=5.0,
        metadata={
            "help": "Contact-force threshold in newtons for contact-aware EMA.",
            "min": 0.0,
        },
    )
    eval_pre_roll_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Before a scored full-motion evaluation, hold reference time at "
                "zero for this many policy/physics steps. This establishes contact "
                "and previous-action history after a terrain-safe reset."
            ),
            "min": 0,
        },
    )
    quality_checkpoint_score: bool = field(
        default=False,
        metadata={
            "help": (
                "Use success minus tracking/smoothness penalties for best-checkpoint "
                "selection. Disabled by default to preserve upstream behavior."
            )
        },
    )
    quality_score_position_metric: str = "gt_error"
    quality_score_success_weight: float = 10.0
    quality_score_gt_weight: float = 1.0
    quality_score_gr_weight: float = 0.25
    quality_score_jerk_weight: float = 1.0e-4
    quality_score_opening_jerk_weight: float = 2.0e-4
    quality_score_action_delta_weight: float = 0.1
