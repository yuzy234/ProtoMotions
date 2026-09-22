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
"""Configuration classes for motion manager components.

This module contains all configuration dataclasses for motion manager functionality,
co-located with the motion manager implementations in the same directory.
"""

from typing import Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class MotionManagerConfig:
    """Configuration for motion management."""

    _target_: str = "protomotions.envs.motion_manager.motion_manager.MotionManager"

    init_start_prob: float = field(
        default=0.2,
        metadata={
            "help": "Probability to sample an initial pose instead of random time. Helps prevent local-minima in AMP.",
            "min": 0.0,
            "max": 1.0,
        }
    )

    subset_method: Optional[Union[str, List[int]]] = field(
        default=None,
        metadata={
            "help": "Motion subset for evaluation: 'first', 'last', 'random', or list of motion IDs. None uses all motions.",
            "options": ["first", "last", "random"],
        }
    )

    exclude_motion_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Motion IDs to exclude from sampling. Useful for removing problematic motions.",
        }
    )

    exclude_motions_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to file with motion IDs to exclude (one per line). Can also be an expert training directory.",
        }
    )

    realign_motion_with_humanoid_on_each_step: bool = field(
        default=False,
        metadata={
            "help": "Realign motion with humanoid each step. Prevents tracking error accumulation for imperfect retargeting.",
        }
    )


@dataclass
class MimicMotionManagerConfig(MotionManagerConfig):
    """Configuration for mimic motion management."""

    _target_: str = (
        "protomotions.envs.motion_manager.mimic_motion_manager.MimicMotionManager"
    )

    resample_on_reset: bool = field(
        default=True,
        metadata={"help": "Whether to resample motion on environment reset."}
    )


@dataclass
class OverlappingClipMotionManagerConfig(MimicMotionManagerConfig):
    """Single-motion curriculum over overlapping fixed-duration tracking clips."""

    _target_: str = (
        "protomotions.envs.motion_manager.overlapping_clip_motion_manager."
        "OverlappingClipMotionManager"
    )
    clip_motion_id: int = field(default=0)
    clip_duration: float = field(default=2.0, metadata={"help": "Clip duration in seconds."})
    clip_stride: float = field(default=1.0, metadata={"help": "Training-window stride in seconds."})
    failure_sampling_mix: float = field(
        default=0.8,
        metadata={"help": "Probability mass from failure-weighted rather than uniform sampling."},
    )
    max_failure_weight_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional upper ratio between the hardest and easiest clip "
                "curriculum weights. None preserves pure failure-rate sampling."
            )
        },
    )
    full_motion_sampling_probability: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of sampling a start-to-finish episode in addition "
                "to overlapping short windows. This exposes accumulated drift "
                "that cannot be observed in independently reset clips."
            )
        },
    )
    record_episode_events: bool = field(
        default=False,
        metadata={
            "help": (
                "Write one CPU-side JSON event per completed environment. "
                "Disabled by default because large vectorized runs can build "
                "millions of Python dictionaries and force GPU synchronization."
            )
        },
    )
