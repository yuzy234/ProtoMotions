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

"""Frozen motion-tracker actor with a trainable reference-root residual.

This module is intentionally implemented as a PPO actor replacement.  It keeps a
pretrained motion-tracker actor frozen, learns a small residual MLP from the same
observations, applies that residual to the reference-root position information in
``mimic_target_poses``, then forwards the corrected observation through the frozen
tracker actor.

The residual is zero-initialized at the final layer, so the initial policy is
exactly the pretrained tracker policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.common.config import MLPWithConcatConfig
from protomotions.agents.ppo.config import PPOActorConfig
from protomotions.agents.ppo.model import PPOActor
from protomotions.utils.hydra_replacement import get_class


@dataclass
class RootResidualTrackerActorConfig(PPOActorConfig):
    """Config for :class:`RootResidualTrackerActor`.

    Args:
        frozen_actor: Original tracker actor config.  This actor is loaded from
            ``frozen_actor_checkpoint`` and frozen.
        residual_model: Trainable MLP config.  It should read the same observation
            keys as the tracker actor and write ``root_pos_residual``.
        frozen_actor_checkpoint: Path to the pretrained tracker ``last.ckpt``.
        mimic_target_pose_key: TensorDict key to rewrite before calling tracker.
        num_future_steps: Number of future target poses packed in
            ``mimic_target_poses``.
        num_bodies: Number of rigid bodies in the max-coords target-pose layout.
        residual_scale: Multiplier for the raw residual MLP output, in meters.
        apply_to_position_blocks: For the standard max-coords-with-velocities
            layout, the first two blocks are position blocks:
            ``target_body_pos`` and ``target_body_pos_rel``.  Applying to both
            corresponds to translating the whole reference pose by the root
            residual.
    """

    _target_: str = "protomotions.agents.mimic.root_residual_tracker_actor.RootResidualTrackerActor"
    frozen_actor: Optional[PPOActorConfig] = None
    residual_model: Optional[MLPWithConcatConfig] = None
    frozen_actor_checkpoint: str = ""
    mimic_target_pose_key: str = "mimic_target_poses"
    residual_key: str = "root_pos_residual"
    scaled_residual_key: str = "root_pos_residual_scaled"
    num_future_steps: int = 5
    num_bodies: int = 24
    residual_scale: float = 0.25
    apply_to_position_blocks: int = 2


class RootResidualTrackerActor(TensorDictModuleBase):
    """Trainable root-residual wrapper around a frozen PPO tracker actor."""

    config: RootResidualTrackerActorConfig

    def __init__(self, config: RootResidualTrackerActorConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        FrozenActorClass = get_class(self.config.frozen_actor._target_)
        self.frozen_actor: PPOActor = FrozenActorClass(config=self.config.frozen_actor)

        ResidualClass = get_class(self.config.residual_model._target_)
        self.residual_model: TensorDictModuleBase = ResidualClass(
            config=self.config.residual_model
        )

        # Expose logstd for PPO.actor_step().  It is copied from the frozen
        # tracker checkpoint when the frozen actor is loaded.
        self.logstd = nn.Parameter(
            torch.ones(self.config.num_out) * self.config.actor_logstd,
            requires_grad=False,
        )

        self.in_keys = list(
            dict.fromkeys(self.config.frozen_actor.in_keys + self.config.residual_model.in_keys)
        )
        self.out_keys = self.config.out_keys

        self._frozen_loaded = False
        self._residual_zero_initialized = False

        # Frozen actor uses LazyLinear.  Parameters cannot be frozen until after
        # lazy materialization; this is done in _load_frozen_actor_from_checkpoint().

    def _zero_init_residual_last_layer(self) -> None:
        """Zero-initialize the last Linear/LazyLinear after lazy materialization."""

        if self._residual_zero_initialized:
            return

        last_linear = None
        for module in self.residual_model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is None:
            raise RuntimeError("Could not find final Linear layer in residual_model.")

        with torch.no_grad():
            last_linear.weight.zero_()
            if last_linear.bias is not None:
                last_linear.bias.zero_()

        self._residual_zero_initialized = True

    def _load_frozen_actor_from_checkpoint(self, tensordict: TensorDict) -> None:
        """Materialize lazy tracker modules, load actor weights, and freeze them."""

        if self._frozen_loaded:
            return

        if not self.config.frozen_actor_checkpoint:
            raise ValueError("frozen_actor_checkpoint must be set.")

        # Materialize LazyLinear/normalizers with the current observation shape.
        with torch.no_grad():
            _ = self.frozen_actor(tensordict.clone())

        checkpoint_path = Path(self.config.frozen_actor_checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Frozen actor checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=tensordict.device,
            weights_only=False,
        )
        model_state = checkpoint["model"]

        actor_state = {}
        prefix = "_actor."
        for key, value in model_state.items():
            if key.startswith(prefix):
                actor_state[key[len(prefix):]] = value

        missing, unexpected = self.frozen_actor.load_state_dict(actor_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading frozen actor: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys while loading frozen actor: {missing}")

        with torch.no_grad():
            self.logstd.copy_(self.frozen_actor.logstd.detach())

        self.frozen_actor.eval()
        for parameter in self.frozen_actor.parameters():
            parameter.requires_grad_(False)

        self._frozen_loaded = True

    def _apply_root_residual(self, target_poses: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply heading-local root residual to max-coords target-pose position blocks."""

        if target_poses.ndim != 2:
            raise ValueError(
                f"Expected flattened target poses [N, F], got shape {tuple(target_poses.shape)}"
            )

        if target_poses.shape[-1] % self.config.num_future_steps != 0:
            raise ValueError(
                f"{self.config.mimic_target_pose_key} dim {target_poses.shape[-1]} is not divisible "
                f"by num_future_steps={self.config.num_future_steps}"
            )

        batch_size = target_poses.shape[0]
        per_step_dim = target_poses.shape[-1] // self.config.num_future_steps
        pos_block_dim = self.config.num_bodies * 3
        required_dim = self.config.apply_to_position_blocks * pos_block_dim
        if per_step_dim < required_dim:
            raise ValueError(
                f"Target-pose per-step dim {per_step_dim} is too small for "
                f"{self.config.apply_to_position_blocks} position blocks with "
                f"num_bodies={self.config.num_bodies}."
            )

        corrected = target_poses.view(batch_size, self.config.num_future_steps, per_step_dim).clone()
        residual = residual.view(batch_size, 1, 1, 3) * self.config.residual_scale

        for block_idx in range(self.config.apply_to_position_blocks):
            start = block_idx * pos_block_dim
            stop = start + pos_block_dim
            pos_block = corrected[..., start:stop].view(
                batch_size, self.config.num_future_steps, self.config.num_bodies, 3
            )
            pos_block = pos_block + residual
            corrected[..., start:stop] = pos_block.reshape(
                batch_size, self.config.num_future_steps, pos_block_dim
            )

        return corrected.view_as(target_poses)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Compute residual-corrected frozen tracker action."""

        # First residual forward materializes the LazyLinear layers.
        tensordict = self.residual_model(tensordict)
        self._zero_init_residual_last_layer()

        # If we just zero-initialized after materialization, recompute so the
        # first real output is exactly zero.
        if torch.any(tensordict[self.config.residual_key] != 0):
            tensordict = self.residual_model(tensordict)

        residual = tensordict[self.config.residual_key]
        if residual.shape[-1] != 3:
            raise ValueError(
                f"Root residual must be 3D, got shape {tuple(residual.shape)}"
            )
        scaled_residual = residual * self.config.residual_scale
        tensordict[self.config.scaled_residual_key] = scaled_residual

        corrected_td = tensordict.clone()
        corrected_td[self.config.mimic_target_pose_key] = self._apply_root_residual(
            tensordict[self.config.mimic_target_pose_key], residual
        )

        self._load_frozen_actor_from_checkpoint(corrected_td)

        corrected_td = self.frozen_actor(corrected_td)
        for key in self.config.out_keys:
            tensordict[key] = corrected_td[key]

        return tensordict
