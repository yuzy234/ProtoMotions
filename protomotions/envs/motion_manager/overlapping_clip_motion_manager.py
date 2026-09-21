# SPDX-License-Identifier: Apache-2.0
"""Failure-weighted sampling of overlapping windows from one reference motion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from protomotions.components.motion_lib import MotionLib
from protomotions.envs.motion_manager.config import OverlappingClipMotionManagerConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager


class OverlappingClipMotionManager(MimicMotionManager):
    """Treat overlapping windows of one motion as curriculum sampling units.

    A normal clip-end is a truncation (not a failure).  Only environment
    terminations recorded through :meth:`record_episode_outcomes` count as clip
    failures.  Evaluation can disable clip limits and then runs the full motion.
    """

    config: OverlappingClipMotionManagerConfig

    def __init__(
        self,
        config: OverlappingClipMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
    ):
        super().__init__(config, num_envs, env_dt, device, motion_lib, fixed_motion_ids_per_env)
        if config.clip_motion_id < 0 or config.clip_motion_id >= motion_lib.num_motions():
            raise ValueError(f"clip_motion_id={config.clip_motion_id} is outside motion library.")
        if config.clip_duration <= 0 or config.clip_stride <= 0:
            raise ValueError("clip_duration and clip_stride must be positive.")
        max_failure_weight_ratio = getattr(config, "max_failure_weight_ratio", None)
        if max_failure_weight_ratio is not None and max_failure_weight_ratio < 1.0:
            raise ValueError("max_failure_weight_ratio must be >= 1 or None.")
        full_motion_probability = float(
            getattr(config, "full_motion_sampling_probability", 0.0)
        )
        if not 0.0 <= full_motion_probability <= 1.0:
            raise ValueError("full_motion_sampling_probability must lie in [0, 1].")

        motion_length = float(motion_lib.motion_lengths[config.clip_motion_id].item())
        if motion_length <= env_dt:
            raise ValueError("Selected motion is shorter than one control step.")
        # MotionLib is defined at its final timestamp, and future observations
        # clamp there.  Keep the final full-duration window instead of dropping
        # one control step from it.
        max_start = max(0.0, motion_length - config.clip_duration)
        starts = torch.arange(0.0, max_start + 1e-6, config.clip_stride, device=device)
        if starts.numel() == 0:
            starts = torch.zeros(1, device=device)
        if max_start > 1e-6 and starts[-1] < max_start - 1e-6:
            starts = torch.cat((starts, torch.tensor([max_start], device=device)))

        ends = torch.minimum(
            starts + config.clip_duration,
            torch.full_like(starts, motion_length),
        )
        self.full_motion_clip_id: Optional[int] = None
        if full_motion_probability > 0.0:
            duplicate = torch.nonzero(
                (starts.abs() <= 1e-6) & ((ends - motion_length).abs() <= 1e-6),
                as_tuple=False,
            ).flatten()
            if duplicate.numel() > 0:
                self.full_motion_clip_id = int(duplicate[0].item())
            else:
                self.full_motion_clip_id = int(starts.numel())
                starts = torch.cat((starts, torch.zeros(1, device=device)))
                ends = torch.cat(
                    (ends, torch.tensor([motion_length], device=device))
                )

        self.clip_starts = starts
        self.clip_ends = ends
        self.clip_attempts = torch.zeros(starts.numel(), dtype=torch.long, device=device)
        self.clip_failures = torch.zeros(starts.numel(), dtype=torch.long, device=device)
        self.current_clip_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_clip_end_times = torch.zeros(num_envs, device=device)
        self.clip_mode_enabled = True
        self.episode_counter = 0
        self._pending_episode_events = []
        print(
            f"Overlapping clip curriculum: {starts.numel()} windows, "
            f"duration={config.clip_duration:.2f}s, stride={config.clip_stride:.2f}s, "
            f"full_motion_probability={full_motion_probability:.2f}"
        )

    def set_clip_mode(self, enabled: bool) -> None:
        """Enable training windows or full-motion evaluation playback."""
        self.clip_mode_enabled = enabled

    def _sampling_probabilities(self) -> torch.Tensor:
        # Beta(1, 1) smoothing: unseen windows start at 0.5 rather than being
        # treated as certain successes or failures.
        failure_rate = (self.clip_failures.float() + 1.0) / (self.clip_attempts.float() + 2.0)
        max_ratio = getattr(self.config, "max_failure_weight_ratio", None)
        if max_ratio is None:
            failure_weights = failure_rate
        else:
            # A very high cold-start failure rate can otherwise monopolize the
            # rollout budget even when full-motion deployment never starts at
            # that state. Keep hard-example emphasis, but bound the hardest to
            # at most R times the easiest curriculum weight.
            normalized_difficulty = failure_rate / failure_rate.max().clamp(min=1e-8)
            ratio = float(max_ratio)
            bounded_weights = 1.0 + (ratio - 1.0) * normalized_difficulty
            failure_weights = bounded_weights

        full_probability = float(
            getattr(self.config, "full_motion_sampling_probability", 0.0)
        )
        if self.full_motion_clip_id is None:
            eligible = torch.ones_like(failure_rate, dtype=torch.bool)
        else:
            eligible = torch.ones_like(failure_rate, dtype=torch.bool)
            eligible[self.full_motion_clip_id] = False
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return torch.ones_like(failure_rate)

        failure_prob = torch.zeros_like(failure_rate)
        failure_prob[eligible] = (
            failure_weights[eligible] / failure_weights[eligible].sum()
        )
        uniform_prob = torch.zeros_like(failure_rate)
        uniform_prob[eligible] = 1.0 / eligible_count
        mix = self.config.failure_sampling_mix
        probabilities = mix * failure_prob + (1.0 - mix) * uniform_prob
        if self.full_motion_clip_id is not None:
            probabilities *= 1.0 - full_probability
            probabilities[self.full_motion_clip_id] = full_probability
        return probabilities

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ) -> None:
        if not self.clip_mode_enabled:
            return super().sample_motions(env_ids, new_motion_ids)
        del new_motion_ids
        clip_ids = torch.multinomial(self._sampling_probabilities(), len(env_ids), replacement=True)
        self.motion_ids[env_ids] = self.config.clip_motion_id
        self.motion_times[env_ids] = self.clip_starts[clip_ids]
        self.current_clip_ids[env_ids] = clip_ids
        self.current_clip_end_times[env_ids] = self.clip_ends[clip_ids]

    def get_done_tracks(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.clip_mode_enabled:
            # Repeated float32 additions of env_dt can land below a mathematically
            # exact clip end; half a control step is safely smaller than one
            # rollout transition while covering accumulated rounding error.
            done = self.motion_times >= (
                self.current_clip_end_times - 0.5 * self.env_dt
            )
        else:
            done = super().get_done_tracks()
        return done if env_ids is None else done[env_ids]

    def record_episode_outcomes(
        self, dones: torch.Tensor, terminated: torch.Tensor) -> dict:
        """Update clip attempts/failures after a simulation step.

        ``terminated`` comes from tracking termination; a normal clip boundary
        has ``done=True`` and ``terminated=False`` and is counted as success.
        """
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0 and self.clip_mode_enabled:
            clip_ids = self.current_clip_ids[done_ids]
            self.clip_attempts.scatter_add_(0, clip_ids, torch.ones_like(clip_ids))
            failure_values = terminated[done_ids].to(dtype=torch.long)
            self.clip_failures.scatter_add_(0, clip_ids, failure_values)
            # This happens only at episode ends (normally every 60 steps), so
            # the small CPU transfer is acceptable and gives an auditable map
            # from episode/env to the sampled clip.
            elapsed_steps = torch.round(
                (self.motion_times[done_ids] - self.clip_starts[clip_ids]) / self.env_dt
            ).long()
            for env_id, clip_id, failed, steps in zip(
                done_ids.detach().cpu().tolist(),
                clip_ids.detach().cpu().tolist(),
                failure_values.detach().cpu().tolist(),
                elapsed_steps.detach().cpu().tolist(),
            ):
                self._pending_episode_events.append(
                    {
                        "episode_id": self.episode_counter,
                        "env_id": env_id,
                        "clip_id": clip_id,
                        "clip_start_time": float(self.clip_starts[clip_id].item()),
                        "failed": bool(failed),
                        "steps": int(steps),
                    }
                )
                self.episode_counter += 1

        rates = (self.clip_failures.float() + 1.0) / (self.clip_attempts.float() + 2.0)
        hardest = torch.argmax(rates)
        return {
            "clip/current_id_mean": self.current_clip_ids.float().mean(),
            "clip/current_start_s_mean": self.clip_starts[self.current_clip_ids].mean(),
            "clip/failure_rate_mean": rates.mean(),
            "clip/failure_rate_max": rates.max(),
            "clip/hardest_id": hardest.float(),
            "clip/hardest_start_s": self.clip_starts[hardest],
            "clip/attempts_total": self.clip_attempts.sum().float(),
            "clip/failures_total": self.clip_failures.sum().float(),
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state.update(
            {
                "clip_starts": self.clip_starts.cpu(),
                "clip_attempts": self.clip_attempts.cpu(),
                "clip_failures": self.clip_failures.cpu(),
                "episode_counter": self.episode_counter,
            }
        )
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "clip_starts" not in state_dict:
            return
        saved_starts = state_dict["clip_starts"].to(self.device)
        if saved_starts.shape != self.clip_starts.shape or not torch.allclose(saved_starts, self.clip_starts):
            print("Warning: clip curriculum state does not match current motion/config; using fresh statistics.")
            return
        self.clip_attempts.copy_(state_dict["clip_attempts"].to(self.device))
        self.clip_failures.copy_(state_dict["clip_failures"].to(self.device))
        self.episode_counter = int(state_dict.get("episode_counter", 0))

    def save_clip_statistics(self, path: Path) -> None:
        rates = (self.clip_failures.float() + 1.0) / (self.clip_attempts.float() + 2.0)
        payload = {
            "motion_id": self.config.clip_motion_id,
            "clip_duration_seconds": self.config.clip_duration,
            "clip_stride_seconds": self.config.clip_stride,
            "clips": [
                {
                    "clip_id": int(i),
                    "start_time": float(self.clip_starts[i].item()),
                    "end_time": float(self.clip_ends[i].item()),
                    "duration_seconds": float(
                        (self.clip_ends[i] - self.clip_starts[i]).item()
                    ),
                    "is_full_motion": i == self.full_motion_clip_id,
                    "attempts": int(self.clip_attempts[i].item()),
                    "failures": int(self.clip_failures[i].item()),
                    "failure_rate_smoothed": float(rates[i].item()),
                }
                for i in range(self.clip_starts.numel())
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        if self._pending_episode_events:
            events_path = path.with_name("clip_episode_events.jsonl")
            with events_path.open("a", encoding="utf-8") as handle:
                for event in self._pending_episode_events:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._pending_episode_events.clear()
