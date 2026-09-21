# SPDX-License-Identifier: Apache-2.0
"""Frozen full-body tracker with a trainable raw-Gaussian action residual."""

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
class ActionResidualTrackerActorConfig(PPOActorConfig):
    """Configuration for :class:`ActionResidualTrackerActor`.

    ``actor_logstd`` is the residual latent standard deviation, *before*
    multiplication by ``residual_noise_scale``.  The final policy is

    ``N(base_mu_raw + mean_scale * tanh(residual_mu),
    (noise_scale * sigma)^2)``.

    The surrounding ProtoMotions environment applies the sole ``tanh`` before
    converting raw actions into PD targets.  Keeping that interface is what
    makes zero residual exactly reproduce the pretrained tracker.
    """

    _target_: str = (
        "protomotions.agents.mimic.action_residual_tracker_actor."
        "ActionResidualTrackerActor"
    )
    frozen_actor: Optional[PPOActorConfig] = None
    residual_model: Optional[MLPWithConcatConfig] = None
    frozen_actor_checkpoint: str = ""
    base_mu_key: str = "base_mu_raw"
    residual_key: str = "residual_action_raw"
    raw_mean_key: str = "raw_mean_action"
    residual_mean_scale: float = 1.0
    residual_noise_scale: float = 0.2
    min_logstd: float = -3.0
    max_logstd: float = -0.5
    disable_residual: bool = False


class ActionResidualTrackerActor(PPOActor):
    """PPO actor whose policy is a residual around a frozen tracker mean.

    The frozen tracker is evaluated deterministically.  Only the residual MLP and
    residual log standard deviation are trainable.  The actor deliberately
    returns raw Gaussian actions because the environment owns the action tanh,
    exactly as in the pretrained tracker.
    """

    config: ActionResidualTrackerActorConfig

    def __init__(self, config: ActionResidualTrackerActorConfig):
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
        # base_mu_key is produced internally from the frozen actor, so it is
        # deliberately not an external PPO/environment observation key.
        self.in_keys = list(config.frozen_actor.in_keys)
        self.out_keys = config.out_keys
        self._frozen_loaded = False
        self._residual_zero_initialized = False

    def train(self, mode: bool = True):
        """Keep the pretrained actor, including its observation normalizer, frozen."""
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

        # Materialize lazy layers before loading their checkpoint tensors. Keep
        # its normalizer in eval mode even for this one dummy forward.
        self.frozen_actor.eval()
        with torch.no_grad():
            self.frozen_actor(tensordict.clone())

        checkpoint_path = Path(self.config.frozen_actor_checkpoint).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location=tensordict.device, weights_only=False)
        actor_state = {
            key[len("_actor.") :]: value
            for key, value in checkpoint["model"].items()
            if key.startswith("_actor.")
        }
        missing, unexpected = self.frozen_actor.load_state_dict(actor_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Could not load frozen actor (missing={missing}, unexpected={unexpected})."
            )
        self.frozen_actor.eval()
        for parameter in self.frozen_actor.parameters():
            parameter.requires_grad_(False)
        self._frozen_loaded = True

    def _effective_std(self) -> torch.Tensor:
        logstd = self.logstd.clamp(self.config.min_logstd, self.config.max_logstd)
        noise_scale = getattr(
            self.config,
            "residual_noise_scale",
            getattr(self.config, "residual_scale", 0.2),
        )
        return torch.exp(logstd) * noise_scale

    @staticmethod
    def _raw_neglogp(action: torch.Tensor, raw_mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        dist = Normal(raw_mean, raw_mean * 0.0 + std)
        return -dist.log_prob(action).sum(dim=-1)

    def neglogp_from_actions(self, action: torch.Tensor, tensordict: TensorDict) -> torch.Tensor:
        """Evaluate the raw Gaussian density used by PPO."""
        return self._raw_neglogp(
            action,
            tensordict[self.config.raw_mean_key],
            self._effective_std(),
        )

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # The frozen actor's mu network yields the pre-tanh action mean without
        # sampling stochastic base actions.
        self._load_frozen_actor(tensordict)
        with torch.no_grad():
            base_td = self.frozen_actor.mu(tensordict.clone())
            base_mu = base_td[self.config.frozen_actor.mu_key]
        tensordict[self.config.base_mu_key] = base_mu

        needs_zero_initialization = not self._residual_zero_initialized
        tensordict = self.residual_model(tensordict)
        if needs_zero_initialization:
            self._zero_init_residual_last_layer()
            # The first LazyLinear materialization happened before zeroing.
            tensordict = self.residual_model(tensordict)

        residual_mu = tensordict[self.config.residual_key]
        if getattr(self.config, "disable_residual", False):
            residual_mu = torch.zeros_like(residual_mu)
        bounded_residual_mu = torch.tanh(residual_mu)
        mean_scale = getattr(
            self.config,
            "residual_mean_scale",
            getattr(self.config, "residual_scale", 0.2),
        )
        residual_mean = mean_scale * bounded_residual_mu
        raw_mean = base_mu + residual_mean
        std = self._effective_std()
        raw_action = Normal(raw_mean, raw_mean * 0.0 + std).sample()

        tensordict[self.config.raw_mean_key] = raw_mean
        # Exposed for diagnostics; it is not an environment observation.
        tensordict["residual_action_mean"] = residual_mean
        tensordict["action"] = raw_action
        tensordict["mean_action"] = raw_mean
        tensordict["neglogp"] = self._raw_neglogp(raw_action, raw_mean, std)
        return tensordict
