# SPDX-License-Identifier: Apache-2.0
"""Frozen full-body tracker with a trainable penultimate-latent residual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal
from tensordict import TensorDict

from protomotions.agents.common.config import MLPWithConcatConfig
from protomotions.agents.ppo.config import PPOActorConfig
from protomotions.agents.ppo.model import PPOActor
from protomotions.utils.hydra_replacement import get_class


@dataclass
class LatentResidualTrackerActorConfig(PPOActorConfig):
    """Configuration for a residual adapter in the frozen actor's latent.

    For the SMPL terrain tracker, ``base_latent`` is the 1024-dimensional input
    to the actor's final 69-dimensional linear action head.  The policy mean is

    ``action_head(base_latent + scale * tanh(latent_residual))``.

    PPO still models a Gaussian in the final raw-action space.  The environment
    applies the only action ``tanh`` before mapping actions to PD targets.
    """

    _target_: str = (
        "protomotions.agents.mimic.latent_residual_tracker_actor."
        "LatentResidualTrackerActor"
    )
    frozen_actor: Optional[PPOActorConfig] = None
    residual_model: Optional[MLPWithConcatConfig] = None
    frozen_actor_checkpoint: str = ""
    base_latent_key: str = "base_actor_latent"
    residual_key: str = "latent_residual_raw"
    raw_mean_key: str = "raw_mean_action"
    latent_residual_scale: float = 0.25
    residual_noise_scale: float = 0.2
    min_logstd: float = -3.0
    max_logstd: float = -0.5
    disable_residual: bool = False
    residual_warmup_key: Optional[str] = None


class LatentResidualTrackerActor(PPOActor):
    """PPO actor that adapts a frozen tracker's penultimate hidden feature.

    The pretrained observation normalizer, six-layer MLP trunk, and final action
    head are all frozen.  Gradients pass through the frozen action head into the
    trainable latent adapter, but never update the pretrained parameters.
    """

    config: LatentResidualTrackerActorConfig

    def __init__(self, config: LatentResidualTrackerActorConfig):
        nn.Module.__init__(self)
        self.config = config
        if config.frozen_actor is None or config.residual_model is None:
            raise ValueError("frozen_actor and residual_model must be configured.")

        FrozenActorClass = get_class(config.frozen_actor._target_)
        self.frozen_actor: PPOActor = FrozenActorClass(config=config.frozen_actor)
        ResidualClass = get_class(config.residual_model._target_)
        self.residual_model = ResidualClass(config=config.residual_model)

        self.logstd = nn.Parameter(
            torch.ones(config.num_out) * config.actor_logstd,
            requires_grad=config.learnable_std,
        )
        residual_obs_keys = [
            key
            for key in config.residual_model.in_keys
            if key != config.base_latent_key
        ]
        self.in_keys = list(
            dict.fromkeys(list(config.frozen_actor.in_keys) + residual_obs_keys)
        )
        self.out_keys = config.out_keys
        self._frozen_loaded = False
        self._residual_zero_initialized = False
        self._action_head: Optional[nn.Linear] = None

    def train(self, mode: bool = True):
        """Keep every pretrained tracker module and normalizer in eval mode."""
        super().train(mode)
        self.frozen_actor.eval()
        return self

    def _zero_init_residual_last_layer(self) -> None:
        if self._residual_zero_initialized:
            return
        last_linear = None
        for module in self.residual_model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise RuntimeError("Could not find the residual model final Linear layer.")
        with torch.no_grad():
            last_linear.weight.zero_()
            if last_linear.bias is not None:
                last_linear.bias.zero_()
        self._residual_zero_initialized = True

    def _load_frozen_actor(self, tensordict: TensorDict) -> None:
        if self._frozen_loaded:
            return
        if not self.config.frozen_actor_checkpoint:
            raise ValueError("frozen_actor_checkpoint must be set.")

        self.frozen_actor.eval()
        with torch.no_grad():
            self.frozen_actor(tensordict.clone())

        checkpoint_path = Path(self.config.frozen_actor_checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Frozen actor checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=tensordict.device, weights_only=False
        )
        actor_state = {
            key[len("_actor.") :]: value
            for key, value in checkpoint["model"].items()
            if key.startswith("_actor.")
        }
        if not actor_state:
            raise RuntimeError(f"No _actor.* keys found in {checkpoint_path}.")
        missing, unexpected = self.frozen_actor.load_state_dict(actor_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Could not load frozen actor (missing={missing}, unexpected={unexpected})."
            )

        mu_mlp = getattr(getattr(self.frozen_actor, "mu", None), "mlp", None)
        if not isinstance(mu_mlp, nn.Sequential) or not isinstance(mu_mlp[-1], nn.Linear):
            raise TypeError(
                "LatentResidualTrackerActor requires frozen_actor.mu.mlp to end "
                "with the Linear action head."
            )
        if mu_mlp[-1].out_features != self.config.num_out:
            raise ValueError(
                "Frozen action-head output size does not match num_out: "
                f"{mu_mlp[-1].out_features} != {self.config.num_out}."
            )
        # Keep a non-registering alias. The head is already registered below
        # frozen_actor.mu.mlp; registering it twice would duplicate checkpoint
        # keys and make resume depend on forward/materialization order.
        object.__setattr__(self, "_action_head", mu_mlp[-1])

        self.frozen_actor.eval()
        for parameter in self.frozen_actor.parameters():
            parameter.requires_grad_(False)
        self._frozen_loaded = True

    def _extract_base_latent_and_mean(
        self, tensordict: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen actor and capture the input to its final action head."""
        if self._action_head is None:
            raise RuntimeError("Frozen action head has not been initialized.")

        captured: dict[str, torch.Tensor] = {}

        def capture_head_input(_module, args) -> None:
            if len(args) != 1:
                raise RuntimeError("Expected the frozen action head to have one input.")
            captured["latent"] = args[0]

        handle = self._action_head.register_forward_pre_hook(capture_head_input)
        try:
            with torch.no_grad():
                base_td = self.frozen_actor.mu(tensordict.clone())
                base_mean = base_td[self.config.frozen_actor.mu_key]
        finally:
            handle.remove()

        if "latent" not in captured:
            raise RuntimeError("Failed to capture the frozen actor penultimate latent.")
        return captured["latent"].detach(), base_mean.detach()

    def _effective_std(self) -> torch.Tensor:
        logstd = self.logstd.clamp(self.config.min_logstd, self.config.max_logstd)
        return torch.exp(logstd) * self.config.residual_noise_scale

    @staticmethod
    def _raw_neglogp(
        action: torch.Tensor, raw_mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        dist = Normal(raw_mean, raw_mean * 0.0 + std)
        return -dist.log_prob(action).sum(dim=-1)

    def neglogp_from_actions(
        self, action: torch.Tensor, tensordict: TensorDict
    ) -> torch.Tensor:
        return self._raw_neglogp(
            action,
            tensordict[self.config.raw_mean_key],
            self._effective_std(),
        )

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self._load_frozen_actor(tensordict)
        base_latent, base_mean = self._extract_base_latent_and_mean(tensordict)
        tensordict[self.config.base_latent_key] = base_latent

        needs_zero_initialization = not self._residual_zero_initialized
        tensordict = self.residual_model(tensordict)
        if needs_zero_initialization:
            # LazyLinear parameters only exist after the first adapter forward.
            self._zero_init_residual_last_layer()
            tensordict = self.residual_model(tensordict)

        residual_raw = tensordict[self.config.residual_key]
        if residual_raw.shape[-1] != base_latent.shape[-1]:
            raise ValueError(
                "Latent residual width does not match frozen actor latent width: "
                f"{residual_raw.shape[-1]} != {base_latent.shape[-1]}."
            )
        if self.config.disable_residual:
            residual_raw = torch.zeros_like(residual_raw)

        latent_residual = self.config.latent_residual_scale * torch.tanh(residual_raw)
        if self.config.residual_warmup_key is not None:
            warmup = tensordict[self.config.residual_warmup_key]
            if warmup.ndim == latent_residual.ndim - 1:
                warmup = warmup.unsqueeze(-1)
            latent_residual = latent_residual * warmup.clamp(0.0, 1.0)
        corrected_latent = base_latent + latent_residual
        if self._action_head is None:
            raise RuntimeError("Frozen action head has not been initialized.")
        raw_mean = self._action_head(corrected_latent)
        action_residual = raw_mean - base_mean

        std = self._effective_std()
        raw_action = Normal(raw_mean, raw_mean * 0.0 + std).sample()

        tensordict[self.config.raw_mean_key] = raw_mean
        tensordict["base_mu_raw"] = base_mean
        tensordict["latent_residual_mean"] = latent_residual
        tensordict["residual_action_mean"] = action_residual
        tensordict["action"] = raw_action
        tensordict["mean_action"] = raw_mean
        tensordict["neglogp"] = self._raw_neglogp(raw_action, raw_mean, std)
        return tensordict
