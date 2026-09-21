# SPDX-License-Identifier: Apache-2.0
"""Parameter-efficient layerwise adaptation of a frozen motion tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import weakref

import torch
from torch import nn
from torch.distributions import Normal
from tensordict import TensorDict

from protomotions.agents.ppo.config import PPOActorConfig
from protomotions.agents.ppo.model import PPOActor
from protomotions.utils.hydra_replacement import get_class


@dataclass
class LayerwiseAdapterTrackerActorConfig(PPOActorConfig):
    """Insert zero-initialized bottleneck adapters after frozen hidden layers."""

    _target_: str = (
        "protomotions.agents.mimic.layerwise_adapter_tracker_actor."
        "LayerwiseAdapterTrackerActor"
    )
    frozen_actor: Optional[PPOActorConfig] = None
    frozen_actor_checkpoint: str = ""
    adapter_layer_indices: list[int] = field(default_factory=lambda: [3, 5])
    hidden_size: int = 1024
    bottleneck_size: int = 256
    adapter_scale: float = 0.15
    context_key: Optional[str] = None
    reliability_key: Optional[str] = None
    context_bottleneck_size: int = 256
    context_scale: float = 0.0
    action_context_bottleneck_size: int = 0
    action_context_scale: float = 0.0
    action_context_reliability_floor: float = 0.25
    uncertainty_gated_adapters: bool = False
    adapter_uncertainty_floor: float = 0.0
    trainable_tail_layers: int = 0
    pretrained_anchor_coef: float = 0.0
    residual_noise_scale: float = 0.2
    exploration_demand_key: Optional[str] = None
    exploration_demand_max_multiplier: float = 1.0
    exploration_demand_action_indices: list[int] = field(default_factory=list)
    demand_action_bottleneck_size: int = 0
    demand_action_scale: float = 0.0
    demand_action_indices: list[int] = field(default_factory=list)
    raw_mean_key: str = "raw_mean_action"
    min_logstd: float = -3.0
    max_logstd: float = -0.5
    disable_residual: bool = False


def uncertainty_plasticity_gate(
    reliability: torch.Tensor, floor: float = 0.0
) -> torch.Tensor:
    """Allow more policy adaptation where the reference is less reliable."""
    if not 0.0 <= floor <= 1.0:
        raise ValueError("adapter uncertainty floor must lie in [0, 1]")
    return floor + (1.0 - floor) * (1.0 - reliability.clamp(0.0, 1.0))


def demand_conditioned_exploration_std(
    base_std: torch.Tensor,
    demand: torch.Tensor,
    max_multiplier: float,
    action_indices: list[int],
) -> torch.Tensor:
    """Scale selected action dimensions according to a per-sample demand."""
    if base_std.ndim != 1:
        raise ValueError("base_std must have shape [actions]")
    if demand.ndim == 1:
        demand = demand.unsqueeze(-1)
    if demand.ndim != 2 or demand.shape[-1] != 1:
        raise ValueError("demand must have shape [batch] or [batch, 1]")
    if max_multiplier < 1.0:
        raise ValueError("max_multiplier must be at least 1")
    if not action_indices:
        raise ValueError("action_indices cannot be empty")
    if min(action_indices) < 0 or max(action_indices) >= base_std.shape[0]:
        raise ValueError("action_indices contains an out-of-range action index")

    multiplier = base_std.new_ones((demand.shape[0], base_std.shape[0]))
    selected_multiplier = 1.0 + (max_multiplier - 1.0) * demand.clamp(0.0, 1.0)
    multiplier[:, action_indices] = selected_multiplier
    return base_std.unsqueeze(0) * multiplier


def gate_demand_action_residual(
    residual: torch.Tensor,
    demand: torch.Tensor,
    action_indices: list[int],
) -> torch.Tensor:
    """Restrict a learned mean correction to demanded action dimensions."""
    if residual.ndim != 2:
        raise ValueError("residual must have shape [batch, actions]")
    if demand.ndim == 1:
        demand = demand.unsqueeze(-1)
    if demand.shape != (residual.shape[0], 1):
        raise ValueError("demand must have shape [batch] or [batch, 1]")
    if not action_indices:
        raise ValueError("action_indices cannot be empty")
    if min(action_indices) < 0 or max(action_indices) >= residual.shape[1]:
        raise ValueError("action_indices contains an out-of-range action index")
    action_mask = residual.new_zeros(residual.shape[-1])
    action_mask[action_indices] = 1.0
    return residual * action_mask.unsqueeze(0) * demand.clamp(0.0, 1.0)


class _BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int, scale: float):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.scale = scale
        # Exact identity at initialization while allowing gradients into `up`.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.up(self.activation(self.down(hidden)))
        return self.scale * torch.tanh(raw)


class _ContextAdapter(nn.Module):
    """Zero-initialized side path from a temporal reference window."""

    def __init__(self, bottleneck_size: int, hidden_size: int, scale: float):
        super().__init__()
        self.down = nn.LazyLinear(bottleneck_size)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.scale = scale
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.up(self.activation(self.down(context)))
        return self.scale * torch.tanh(raw)


class _ActionContextResidual(nn.Module):
    """Direct action correction from policy features and temporal context.

    Hidden-layer adapters must push a correction through the remainder of the
    frozen policy, which can make their effective action delta extremely small.
    This head keeps the pretrained action unchanged at initialization while
    giving the temporal command window a short, well-conditioned path to the
    final action.
    """

    def __init__(self, bottleneck_size: int, num_out: int, scale: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(bottleneck_size),
            nn.SiLU(),
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.SiLU(),
            nn.Linear(bottleneck_size, num_out),
        )
        self.scale = scale
        output = self.net[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(features))


class LayerwiseAdapterTrackerActor(PPOActor):
    """Frozen six-layer tracker with trainable adapters at selected depths.

    Unlike a single penultimate residual, an early adapter can alter the feature
    computation performed by later frozen layers. The pretrained observation
    normalizer, all original Linear layers, and the action head remain frozen.
    """

    config: LayerwiseAdapterTrackerActorConfig

    def __init__(self, config: LayerwiseAdapterTrackerActorConfig):
        nn.Module.__init__(self)
        self.config = config
        if config.frozen_actor is None:
            raise ValueError("frozen_actor must be configured.")
        if config.uncertainty_gated_adapters and config.reliability_key is None:
            raise ValueError(
                "uncertainty_gated_adapters requires a reliability_key."
            )
        if config.exploration_demand_key is not None:
            if config.exploration_demand_max_multiplier < 1.0:
                raise ValueError(
                    "exploration_demand_max_multiplier must be at least 1."
                )
            if not config.exploration_demand_action_indices:
                raise ValueError(
                    "exploration_demand_action_indices must be configured when "
                    "exploration_demand_key is used."
                )
            if min(config.exploration_demand_action_indices) < 0 or max(
                config.exploration_demand_action_indices
            ) >= config.num_out:
                raise ValueError(
                    "exploration_demand_action_indices contains an invalid action index."
                )
        if config.demand_action_scale > 0.0:
            if config.exploration_demand_key is None:
                raise ValueError(
                    "demand_action_scale requires exploration_demand_key."
                )
            if config.demand_action_bottleneck_size <= 0:
                raise ValueError(
                    "demand_action_bottleneck_size must be positive when the "
                    "demand action expert is enabled."
                )
            if not config.demand_action_indices:
                raise ValueError(
                    "demand_action_indices cannot be empty when the demand action "
                    "expert is enabled."
                )
            if min(config.demand_action_indices) < 0 or max(
                config.demand_action_indices
            ) >= config.num_out:
                raise ValueError("demand_action_indices contains an invalid index.")
        if not config.adapter_layer_indices:
            raise ValueError("At least one adapter layer must be selected.")
        if len(set(config.adapter_layer_indices)) != len(config.adapter_layer_indices):
            raise ValueError("adapter_layer_indices must be unique.")

        FrozenActorClass = get_class(config.frozen_actor._target_)
        self.frozen_actor: PPOActor = FrozenActorClass(config=config.frozen_actor)
        self.adapters = nn.ModuleDict(
            {
                str(index): _BottleneckAdapter(
                    config.hidden_size, config.bottleneck_size, config.adapter_scale
                )
                for index in sorted(config.adapter_layer_indices)
            }
        )
        self.context_adapters = nn.ModuleDict()
        if config.context_key is not None and config.context_scale > 0.0:
            self.context_adapters = nn.ModuleDict(
                {
                    str(index): _ContextAdapter(
                        config.context_bottleneck_size,
                        config.hidden_size,
                        config.context_scale,
                    )
                    for index in sorted(config.adapter_layer_indices)
                }
            )
        self.action_context_residual: Optional[_ActionContextResidual] = None
        if (
            config.context_key is not None
            and config.action_context_bottleneck_size > 0
            and config.action_context_scale > 0.0
        ):
            self.action_context_residual = _ActionContextResidual(
                config.action_context_bottleneck_size,
                config.num_out,
                config.action_context_scale,
            )
        self.demand_action_residual: Optional[_ActionContextResidual] = None
        if config.demand_action_scale > 0.0:
            self.demand_action_residual = _ActionContextResidual(
                config.demand_action_bottleneck_size,
                config.num_out,
                config.demand_action_scale,
            )
        self.logstd = nn.Parameter(
            torch.ones(config.num_out) * config.actor_logstd,
            requires_grad=config.learnable_std,
        )
        self.in_keys = list(config.frozen_actor.in_keys)
        if config.context_key is not None:
            self.in_keys.append(config.context_key)
        if config.reliability_key is not None:
            self.in_keys.append(config.reliability_key)
        if config.exploration_demand_key is not None:
            self.in_keys.append(config.exploration_demand_key)
        self.in_keys = list(dict.fromkeys(self.in_keys))
        self.out_keys = config.out_keys
        self._frozen_loaded = False
        self._adapter_hook_modules: dict[str, nn.Module] = {}
        self._output_linear_ref: Optional[weakref.ReferenceType[nn.Linear]] = None
        self._anchor_parameters: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_actor.eval()
        return self

    def _load_frozen_actor(self, tensordict: TensorDict) -> None:
        if self._frozen_loaded:
            return
        if not self.config.frozen_actor_checkpoint:
            raise ValueError("frozen_actor_checkpoint must be set.")

        self.frozen_actor.eval()
        with torch.no_grad():
            self.frozen_actor(tensordict.clone())
        checkpoint_path = Path(self.config.frozen_actor_checkpoint).expanduser()
        checkpoint = torch.load(
            checkpoint_path, map_location=tensordict.device, weights_only=False
        )
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
        for parameter in self.frozen_actor.parameters():
            parameter.requires_grad_(False)

        mlp = getattr(getattr(self.frozen_actor, "mu", None), "mlp", None)
        if not isinstance(mlp, nn.Sequential):
            raise TypeError("Layerwise adapters require frozen_actor.mu.mlp Sequential.")
        linear_positions = [
            position for position, module in enumerate(mlp) if isinstance(module, nn.Linear)
        ]
        hidden_positions = linear_positions[:-1]
        self._output_linear_ref = weakref.ref(mlp[linear_positions[-1]])
        trainable_tail_layers = int(self.config.trainable_tail_layers)
        if trainable_tail_layers < 0 or trainable_tail_layers > len(hidden_positions):
            raise ValueError(
                "trainable_tail_layers must be in "
                f"[0, {len(hidden_positions)}], got {trainable_tail_layers}."
            )
        if trainable_tail_layers:
            trainable_positions = set(hidden_positions[-trainable_tail_layers:])
            trainable_positions.add(linear_positions[-1])
            for position in trainable_positions:
                for parameter in mlp[position].parameters():
                    parameter.requires_grad_(True)

            # L2-SP anchor: retain a weak pull to the general pretrained
            # tracker while allowing substantially more plasticity than a
            # fully frozen trunk.
            for name, parameter in self.frozen_actor.named_parameters():
                if parameter.requires_grad:
                    self._anchor_parameters[name] = parameter.detach().clone()
        for key in self.adapters:
            layer_index = int(key)
            if layer_index < 0 or layer_index >= len(hidden_positions):
                raise ValueError(
                    f"Adapter layer {layer_index} outside {len(hidden_positions)} hidden layers."
                )
            activation_position = hidden_positions[layer_index] + 1
            if activation_position >= len(mlp) or isinstance(
                mlp[activation_position], nn.Linear
            ):
                raise TypeError(f"No activation follows hidden layer {layer_index}.")
            self._adapter_hook_modules[key] = mlp[activation_position]
        self._frozen_loaded = True

    def _effective_std(self, tensordict: Optional[TensorDict] = None) -> torch.Tensor:
        logstd = self.logstd.clamp(self.config.min_logstd, self.config.max_logstd)
        base_std = torch.exp(logstd) * self.config.residual_noise_scale
        if self.config.exploration_demand_key is None or tensordict is None:
            return base_std
        return demand_conditioned_exploration_std(
            base_std,
            tensordict[self.config.exploration_demand_key],
            self.config.exploration_demand_max_multiplier,
            self.config.exploration_demand_action_indices,
        )

    def effective_std_from_tensordict(self, tensordict: TensorDict) -> torch.Tensor:
        """Return the exact state-conditioned std used by the raw policy."""
        return self._effective_std(tensordict)

    @staticmethod
    def _raw_neglogp(
        action: torch.Tensor, raw_mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return -Normal(raw_mean, raw_mean * 0.0 + std).log_prob(action).sum(-1)

    def neglogp_from_actions(
        self, action: torch.Tensor, tensordict: TensorDict
    ) -> torch.Tensor:
        return self._raw_neglogp(
            action, tensordict[self.config.raw_mean_key], self._effective_std(tensordict)
        )

    def _adapted_mean(
        self, tensordict: TensorDict
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        deltas: list[torch.Tensor] = []
        context_deltas: list[torch.Tensor] = []
        handles = []
        action_context_delta = None
        demand_action_delta = None
        final_hidden: list[torch.Tensor] = []
        context = None
        reliability = None
        if self.config.context_key is not None:
            context = tensordict[self.config.context_key]
        if self.config.reliability_key is not None:
            reliability = tensordict[self.config.reliability_key].clamp(0.0, 1.0)
            if reliability.ndim == 1:
                reliability = reliability.unsqueeze(-1)
        plasticity_gate = None
        if self.config.uncertainty_gated_adapters:
            assert reliability is not None
            plasticity_gate = uncertainty_plasticity_gate(
                reliability, self.config.adapter_uncertainty_floor
            )
        for key, adapter in self.adapters.items():
            module = self._adapter_hook_modules[key]
            context_adapter = (
                self.context_adapters[key] if key in self.context_adapters else None
            )

            def hook(
                _module,
                _args,
                output,
                adapter=adapter,
                context_adapter=context_adapter,
            ):
                delta = adapter(output)
                if plasticity_gate is not None:
                    delta = delta * plasticity_gate
                if context_adapter is not None and context is not None:
                    context_delta = context_adapter(context)
                    if reliability is not None:
                        if plasticity_gate is not None:
                            context_delta = context_delta * plasticity_gate
                        else:
                            context_delta = context_delta * reliability
                    context_deltas.append(context_delta)
                    delta = delta + context_delta
                deltas.append(delta)
                return output + delta

            handles.append(module.register_forward_hook(hook))
        if (
            self.action_context_residual is not None
            or self.demand_action_residual is not None
        ):
            output_linear = (
                None
                if self._output_linear_ref is None
                else self._output_linear_ref()
            )
            if output_linear is None:
                raise RuntimeError("Frozen actor output layer was not initialized.")

            def capture_final_hidden(_module, args):
                final_hidden.append(args[0])

            handles.append(output_linear.register_forward_pre_hook(capture_final_hidden))
        try:
            adapted_td = self.frozen_actor.mu(tensordict.clone())
            mean = adapted_td[self.config.frozen_actor.mu_key]
            if self.action_context_residual is not None:
                if context is None or not final_hidden:
                    raise RuntimeError("Action context residual is missing its inputs.")
                features = [final_hidden[-1], context]
                if reliability is not None:
                    features.append(reliability)
                action_context_delta = self.action_context_residual(
                    torch.cat(features, dim=-1)
                )
                if reliability is not None:
                    if plasticity_gate is not None:
                        gate = plasticity_gate
                    else:
                        floor = float(self.config.action_context_reliability_floor)
                        gate = floor + (1.0 - floor) * reliability
                    action_context_delta = action_context_delta * gate
                mean = mean + action_context_delta
            if self.demand_action_residual is not None:
                if not final_hidden or self.config.exploration_demand_key is None:
                    raise RuntimeError("Demand action expert is missing its inputs.")
                demand = tensordict[self.config.exploration_demand_key]
                if demand.ndim == 1:
                    demand = demand.unsqueeze(-1)
                features = [final_hidden[-1]]
                if context is not None:
                    features.append(context)
                if reliability is not None:
                    features.append(reliability)
                features.append(demand)
                demand_action_delta = self.demand_action_residual(
                    torch.cat(features, dim=-1)
                )
                demand_action_delta = gate_demand_action_residual(
                    demand_action_delta,
                    demand,
                    self.config.demand_action_indices,
                )
                mean = mean + demand_action_delta
        finally:
            for handle in handles:
                handle.remove()
        return (
            mean,
            deltas,
            context_deltas,
            action_context_delta,
            demand_action_delta,
        )

    def pretrained_anchor_loss(self) -> torch.Tensor:
        """Return L2-SP loss for the selectively unfrozen pretrained tail."""
        if not self._anchor_parameters:
            return self.logstd.new_zeros(())
        terms = []
        for name, parameter in self.frozen_actor.named_parameters():
            anchor = self._anchor_parameters.get(name)
            if anchor is not None:
                terms.append((parameter - anchor).square().mean())
        return torch.stack(terms).mean()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self._load_frozen_actor(tensordict)
        with torch.no_grad():
            base_td = self.frozen_actor.mu(tensordict.clone())
            base_mean = base_td[self.config.frozen_actor.mu_key].detach()

        if self.config.disable_residual:
            raw_mean = base_mean
            deltas = [base_mean.new_zeros((*base_mean.shape[:-1], self.config.hidden_size))]
            context_deltas = []
            action_context_delta = None
            demand_action_delta = None
        else:
            (
                raw_mean,
                deltas,
                context_deltas,
                action_context_delta,
                demand_action_delta,
            ) = self._adapted_mean(tensordict)

        action_residual = raw_mean - base_mean
        std = self._effective_std(tensordict)
        raw_action = Normal(raw_mean, raw_mean * 0.0 + std).sample()

        stacked_delta = torch.stack(deltas, dim=-2)
        tensordict[self.config.raw_mean_key] = raw_mean
        tensordict["base_mu_raw"] = base_mean
        tensordict["residual_action_mean"] = action_residual
        tensordict["layer_adapter_delta_rms"] = torch.sqrt(
            stacked_delta.square().mean(dim=(-2, -1))
        )
        if context_deltas:
            stacked_context_delta = torch.stack(context_deltas, dim=-2)
            tensordict["context_adapter_delta_rms"] = torch.sqrt(
                stacked_context_delta.square().mean(dim=(-2, -1))
            )
        if action_context_delta is not None:
            tensordict["action_context_residual_rms"] = torch.sqrt(
                action_context_delta.square().mean(dim=-1)
            )
        if demand_action_delta is not None:
            tensordict["demand_action_residual_rms"] = torch.sqrt(
                demand_action_delta.square().mean(dim=-1)
            )
        if self.config.reliability_key is not None:
            tensordict["reference_reliability_metric"] = tensordict[
                self.config.reliability_key
            ].squeeze(-1)
        expanded_std = raw_mean * 0.0 + std
        tensordict["effective_std_mean"] = expanded_std.mean(dim=-1)
        if self.config.exploration_demand_key is not None:
            tensordict["exploration_demand_metric"] = tensordict[
                self.config.exploration_demand_key
            ].squeeze(-1)
        tensordict["action"] = raw_action
        tensordict["mean_action"] = raw_mean
        tensordict["neglogp"] = self._raw_neglogp(raw_action, raw_mean, std)
        return tensordict
