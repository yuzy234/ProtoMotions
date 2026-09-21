# SPDX-License-Identifier: Apache-2.0
"""PPO model variant that initializes its critic from a pretrained tracker.

This is used by the root-residual experiment: the original tracker actor remains
frozen inside :class:`RootResidualTrackerActor`, the residual actor is trained,
and the critic is initialized from the pretrained motion tracker then finetuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict

from protomotions.agents.ppo.config import PPOModelConfig
from protomotions.agents.ppo.model import PPOModel


@dataclass
class PretrainedCriticPPOModelConfig(PPOModelConfig):
    _target_: str = "protomotions.agents.mimic.root_residual_ppo_model.PretrainedCriticPPOModel"
    critic_checkpoint: str = ""


class PretrainedCriticPPOModel(PPOModel):
    """PPO actor-critic model with one-shot pretrained critic initialization."""

    config: PretrainedCriticPPOModelConfig

    def __init__(self, config: PretrainedCriticPPOModelConfig):
        super().__init__(config)
        self._pretrained_critic_loaded = False

    def _load_pretrained_critic(self, tensordict: TensorDict) -> None:
        if self._pretrained_critic_loaded:
            return

        if not self.config.critic_checkpoint:
            raise ValueError("critic_checkpoint must be set for PretrainedCriticPPOModel.")

        checkpoint_path = Path(self.config.critic_checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained critic checkpoint not found: {checkpoint_path}")

        # Materialize LazyLinear layers and observation-normalizer buffers before
        # loading.  The first dummy value is immediately overwritten by the second
        # critic forward after loading.
        with torch.no_grad():
            self._critic(tensordict)

        checkpoint = torch.load(
            checkpoint_path, map_location=tensordict.device, weights_only=False
        )
        model_state = checkpoint["model"]
        critic_state = {
            key[len("_critic."):]: value
            for key, value in model_state.items()
            if key.startswith("_critic.")
        }
        if not critic_state:
            raise RuntimeError(f"No _critic.* keys found in {checkpoint_path}.")

        missing, unexpected = self._critic.load_state_dict(critic_state, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys while loading pretrained critic: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading pretrained critic: {unexpected}")

        self._pretrained_critic_loaded = True

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._actor(tensordict)
        self._load_pretrained_critic(tensordict)
        tensordict = self._critic(tensordict)
        return tensordict
