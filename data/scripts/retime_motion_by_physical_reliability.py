#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Calibrate noisy MotionLib phase using physical-reference reliability.

Video motion can contain long, physically inconsistent floating intervals even
when the articulated pose sequence is useful.  This utility does not smooth or
alter those poses.  It advances quickly through low-reliability intervals while
keeping reliable support/contact intervals at their recorded rate, then rebuilds
the native robot kinematics and all velocity fields at a uniform output timestep.

The resulting file is an explicit ablation for physics-aware phase calibration;
the source MotionLib remains untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from protomotions.components.pose_lib import (
    compute_angular_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import quaternion_to_matrix, slerp


def _smoothstep01(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def phase_rate_from_reliability(
    reliability: torch.Tensor,
    contacts: torch.Tensor,
    *,
    unreliable_below: float = 0.25,
    reliable_above: float = 0.45,
    max_phase_rate: float = 3.0,
    transition_radius: int = 2,
) -> torch.Tensor:
    """Map physical reliability to a smooth source-phase advancement rate.

    ``1`` retains the observed timing and values greater than one consume the
    corresponding source interval faster.  Any explicit body contact is locked
    to rate one after smoothing so a short contact cannot be erased by adjacent
    unreliable flight frames.
    """
    reliability = torch.as_tensor(reliability, dtype=torch.float64).flatten()
    contacts = torch.as_tensor(contacts, dtype=torch.bool)
    if contacts.ndim == 1:
        contacts = contacts[:, None]
    if contacts.ndim != 2 or contacts.shape[0] != reliability.numel():
        raise ValueError("contacts must have shape [frames, bodies]")
    if reliability.numel() < 2:
        raise ValueError("phase calibration requires at least two frames")
    if not torch.isfinite(reliability).all():
        raise ValueError("reliability contains non-finite values")
    if not 0.0 <= unreliable_below < reliable_above <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= unreliable_below < reliable_above <= 1"
        )
    if max_phase_rate < 1.0:
        raise ValueError("max_phase_rate must be at least one")
    if transition_radius < 0:
        raise ValueError("transition_radius must be non-negative")

    unreliable = (reliable_above - reliability) / (
        reliable_above - unreliable_below
    )
    gate = _smoothstep01(unreliable)
    if transition_radius:
        offsets = torch.arange(
            -transition_radius,
            transition_radius + 1,
            dtype=torch.float64,
        )
        kernel = (transition_radius + 1 - offsets.abs()).clamp(min=0.0)
        kernel /= kernel.sum()
        padded = torch.nn.functional.pad(
            gate[None, None],
            (transition_radius, transition_radius),
            mode="replicate",
        )
        gate = torch.nn.functional.conv1d(
            padded, kernel[None, None]
        ).flatten()

    rate = 1.0 + (max_phase_rate - 1.0) * gate
    rate[contacts.any(dim=-1)] = 1.0
    return rate


def _true_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open runs of True values from a one-dimensional mask."""
    values = torch.as_tensor(mask, dtype=torch.bool).flatten().cpu()
    padded = torch.cat(
        (torch.zeros(1, dtype=torch.bool), values, torch.zeros(1, dtype=torch.bool))
    )
    transitions = torch.nonzero(padded[1:] != padded[:-1]).flatten().tolist()
    return list(zip(transitions[0::2], transitions[1::2]))


def recover_seed_connected_foot_support(
    contacts: torch.Tensor,
    surface_clearance: torch.Tensor,
    surface_speed: torch.Tensor,
    foot_body_ids: list[int] | tuple[int, ...],
    *,
    minimum_clearance_m: float = -0.08,
    maximum_clearance_m: float = 0.42,
    maximum_speed_m_s: float = 1.8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Recover occluded foot support without inventing disconnected contacts.

    Monocular root drift can move a genuinely planted foot several decimetres
    above the reconstructed floor.  A simple distance threshold then truncates
    a stance too early.  We permit a wider *candidate* band, but retain only
    candidate runs connected to at least one original foot-contact seed.  This
    prevents an arbitrary low swinging foot elsewhere in the clip from being
    relabelled as support.

    The selected body is the foot surface with the smallest scene clearance in
    each frame.  Existing labels are never removed.
    """
    contact = torch.as_tensor(contacts, dtype=torch.bool)
    clearance = torch.as_tensor(surface_clearance, dtype=torch.float32)
    speed = torch.as_tensor(surface_speed, dtype=torch.float32)
    if contact.ndim != 2 or clearance.shape != contact.shape or speed.shape != contact.shape:
        raise ValueError("contacts, surface_clearance and surface_speed must match [frames,bodies]")
    if len(contact) < 2:
        raise ValueError("support recovery requires at least two frames")
    ids = torch.as_tensor(foot_body_ids, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("foot_body_ids must be a non-empty one-dimensional sequence")
    if int(ids.min()) < 0 or int(ids.max()) >= contact.shape[1]:
        raise ValueError("foot_body_ids contains an out-of-range body index")
    if minimum_clearance_m >= maximum_clearance_m:
        raise ValueError("minimum_clearance_m must be below maximum_clearance_m")
    if maximum_speed_m_s <= 0:
        raise ValueError("maximum_speed_m_s must be positive")

    foot_clearance = clearance[:, ids]
    selected_local = foot_clearance.argmin(dim=1)
    selected_body = ids[selected_local]
    selected_clearance = foot_clearance.gather(1, selected_local[:, None])[:, 0]
    selected_speed = speed[:, ids].gather(1, selected_local[:, None])[:, 0]
    finite = torch.isfinite(selected_clearance) & torch.isfinite(selected_speed)
    candidate = (
        finite
        & (selected_clearance >= float(minimum_clearance_m))
        & (selected_clearance <= float(maximum_clearance_m))
        & (selected_speed <= float(maximum_speed_m_s))
    )
    observed_foot = contact[:, ids].any(dim=1)
    connected = torch.zeros_like(candidate)
    for start, stop in _true_runs(candidate):
        if observed_foot[start:stop].any():
            connected[start:stop] = True

    recovered = connected & ~contact.any(dim=1)
    augmented = contact.clone()
    recovered_frames = torch.nonzero(recovered).flatten()
    if recovered_frames.numel():
        augmented[recovered_frames, selected_body[recovered_frames]] = True
    return augmented, {
        "candidate": candidate,
        "connected": connected,
        "recovered": recovered,
        "selected_body": selected_body,
        "selected_clearance_m": selected_clearance,
        "selected_speed_m_s": selected_speed,
        "observed_foot_support": observed_foot,
    }


def solve_surface_support_root_correction(
    contacts: torch.Tensor,
    original_contacts: torch.Tensor,
    surface_clearance: torch.Tensor,
    *,
    observation_reliability: torch.Tensor | None = None,
    clearance_lower_m: float = -0.05,
    clearance_upper_m: float = 0.08,
    data_weight: float = 1.0,
    observed_support_weight: float = 80.0,
    recovered_support_weight: float = 50.0,
    correction_jerk_weight: float = 4.0,
    maximum_downward_correction_m: float = 0.60,
    maximum_upward_correction_m: float = 0.30,
) -> torch.Tensor:
    """Solve one smooth vertical gauge correction for surface support.

    The target is the smallest correction that places every labelled support
    surface inside an uncertainty interval, not at one exact height.  Thus a
    hand pressing slightly into a reconstructed wall or a compressed cushion
    is not artificially lifted.  Original contacts receive a stronger weight
    than recovered, occlusion-supported contacts.
    """
    contact = torch.as_tensor(contacts, dtype=torch.bool)
    original = torch.as_tensor(original_contacts, dtype=torch.bool)
    clearance = torch.as_tensor(surface_clearance, dtype=torch.float64)
    if contact.shape != original.shape or clearance.shape != contact.shape:
        raise ValueError("contact and surface-clearance arrays must have identical shapes")
    if clearance_lower_m > clearance_upper_m:
        raise ValueError("clearance interval is reversed")
    if min(data_weight, observed_support_weight, recovered_support_weight, correction_jerk_weight) < 0:
        raise ValueError("least-squares weights must be non-negative")
    if maximum_downward_correction_m < 0 or maximum_upward_correction_m < 0:
        raise ValueError("correction limits must be non-negative")
    frames = contact.shape[0]
    if observation_reliability is None:
        reliability = torch.ones(frames, dtype=torch.float64)
    else:
        reliability = torch.as_tensor(observation_reliability, dtype=torch.float64).flatten()
        if reliability.shape != (frames,):
            raise ValueError("observation_reliability must have one value per frame")
        reliability = reliability.clamp(0.0, 1.0)

    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    def add(indices: tuple[int, ...], coefficients: tuple[float, ...], target: float, weight: float) -> None:
        if weight <= 0:
            return
        row = torch.zeros(frames, dtype=torch.float64)
        row[list(indices)] = torch.as_tensor(coefficients, dtype=torch.float64)
        scale = float(weight) ** 0.5
        rows.append(row * scale)
        targets.append(torch.tensor(float(target) * scale, dtype=torch.float64))

    # Strong image evidence resists movement; uncertain observations retain a
    # non-zero tether so support recovery cannot drift without bound.
    for frame in range(frames):
        visual_weight = float(data_weight) * (0.25 + 0.75 * float(reliability[frame]))
        add((frame,), (1.0,), 0.0, visual_weight)
        body_ids = torch.nonzero(contact[frame]).flatten()
        if body_ids.numel() == 0:
            continue
        values = clearance[frame, body_ids]
        finite_values = values[torch.isfinite(values)]
        if finite_values.numel() == 0:
            continue
        corrections = torch.where(
            finite_values < float(clearance_lower_m),
            float(clearance_lower_m) - finite_values,
            torch.where(
                finite_values > float(clearance_upper_m),
                float(clearance_upper_m) - finite_values,
                torch.zeros_like(finite_values),
            ),
        )
        target = float(corrections.median())
        weight = (
            float(observed_support_weight)
            if original[frame].any()
            else float(recovered_support_weight)
        )
        add((frame,), (1.0,), target, weight)

    for frame in range(frames - 3):
        add(
            (frame, frame + 1, frame + 2, frame + 3),
            (-1.0, 3.0, -3.0, 1.0),
            0.0,
            float(correction_jerk_weight),
        )
    matrix = torch.stack(rows)
    rhs = torch.stack(targets)
    solution = torch.linalg.lstsq(matrix, rhs).solution
    return solution.clamp(
        min=-float(maximum_downward_correction_m),
        max=float(maximum_upward_correction_m),
    ).to(torch.float32)


def reachability_aware_interval_phase_rates(
    center_of_mass: torch.Tensor,
    contacts: torch.Tensor,
    *,
    source_dt: float,
    gravity: float = 9.81,
    max_phase_rate: float = 3.0,
    max_vertical_speed_m_s: float = 6.0,
    max_horizontal_speed_m_s: float = 8.0,
    candidate_count: int = 512,
    compression_penalty: float = 0.05,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Choose a reachable duration for every contact-free interval.

    Slow-motion video can make a person appear to hover.  Conversely, blindly
    compressing every unreliable frame can demand an unrealistically abrupt
    launch.  For each support-bounded free span we search durations between the
    observed clock and ``max_phase_rate`` compression, score the exact ballistic
    endpoint velocities, and use one constant rate for that span.  Support-to-
    support intervals remain at their observed rate.
    """
    com = torch.as_tensor(center_of_mass, dtype=torch.float64)
    contact = torch.as_tensor(contacts, dtype=torch.bool)
    if com.ndim != 2 or com.shape[1] != 3:
        raise ValueError("center_of_mass must have shape [frames,3]")
    if contact.ndim == 1:
        contact = contact[:, None]
    if contact.ndim != 2 or contact.shape[0] != len(com):
        raise ValueError("contacts must have shape [frames,bodies]")
    if source_dt <= 0 or gravity <= 0 or max_phase_rate < 1.0:
        raise ValueError("invalid timing or gravity parameters")
    if min(max_vertical_speed_m_s, max_horizontal_speed_m_s) <= 0:
        raise ValueError("speed limits must be positive")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if len(com) < 2:
        raise ValueError("at least two frames are required")

    support = contact.any(dim=-1)
    interval_rates = torch.ones(len(com) - 1, dtype=torch.float64)
    reports: list[dict[str, Any]] = []
    for run_start, run_stop in _true_runs(~support):
        anchor_start = run_start - 1 if run_start > 0 else run_start
        anchor_stop = run_stop if run_stop < len(com) else run_stop - 1
        if anchor_stop <= anchor_start:
            continue
        source_duration = (anchor_stop - anchor_start) * float(source_dt)
        minimum_duration = source_duration / float(max_phase_rate)
        durations = torch.linspace(
            minimum_duration,
            source_duration,
            candidate_count,
            dtype=torch.float64,
        )
        displacement = com[anchor_stop] - com[anchor_start]
        horizontal_speed = torch.linalg.vector_norm(displacement[:2]) / durations
        initial_vertical_speed = (
            displacement[2] / durations + 0.5 * float(gravity) * durations
        )
        final_vertical_speed = initial_vertical_speed - float(gravity) * durations
        vertical_demand = torch.maximum(
            initial_vertical_speed.abs(), final_vertical_speed.abs()
        )
        phase_rate = source_duration / durations
        score = (
            vertical_demand
            + 0.20 * horizontal_speed
            + float(compression_penalty) * (phase_rate - 1.0)
            + 20.0
            * torch.relu(vertical_demand - float(max_vertical_speed_m_s)).square()
            + 20.0
            * torch.relu(horizontal_speed - float(max_horizontal_speed_m_s)).square()
        )
        best = int(score.argmin())
        selected_rate = float(phase_rate[best])
        interval_rates[anchor_start:anchor_stop] = selected_rate
        reports.append(
            {
                "start_frame": int(anchor_start),
                "end_frame": int(anchor_stop),
                "start_is_support": bool(support[anchor_start]),
                "end_is_support": bool(support[anchor_stop]),
                "source_duration_s": float(source_duration),
                "selected_duration_s": float(durations[best]),
                "phase_rate": selected_rate,
                "horizontal_speed_m_s": float(horizontal_speed[best]),
                "initial_vertical_speed_m_s": float(initial_vertical_speed[best]),
                "final_vertical_speed_m_s": float(final_vertical_speed[best]),
                "within_speed_limits": bool(
                    vertical_demand[best] <= float(max_vertical_speed_m_s)
                    and horizontal_speed[best] <= float(max_horizontal_speed_m_s)
                ),
            }
        )
    return interval_rates, reports


def propagate_ballistic_takeoff_phase_rates(
    interval_phase_rate: torch.Tensor,
    center_of_mass: torch.Tensor,
    contacts: torch.Tensor,
    flight_reports: list[dict[str, Any]],
    *,
    source_dt: float,
    preparation_duration_s: float = 0.40,
    velocity_estimation_duration_s: float = 0.14,
    max_phase_rate: float = 4.0,
    minimum_observed_upward_speed_m_s: float = 0.10,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Propagate a slow-motion flight clock into its takeoff preparation.

    A ballistic projection fixes free-flight geometry, but accelerating only
    the contact-free interval creates an impossible velocity discontinuity at
    takeoff.  The source articulation often already contains the correct
    crouch-and-extension geometry, merely recorded in slow motion.  For each
    support-bounded ascending flight, estimate the source COM launch speed and
    smoothly raise the phase rate over the immediately preceding contiguous
    support run so its terminal speed matches the ballistic launch demand.
    From the last crouch minimum to takeoff, source-frame timestamps are placed
    using ``v^2 = 2 a delta_h``.  This approximates constant vertical COM
    acceleration over the available leg-extension distance instead of creating
    a one-frame velocity step.

    This changes timing only.  It neither edits the articulated pose path nor
    invents a support disconnected from observed/recovered contact evidence.
    """
    rates = torch.as_tensor(interval_phase_rate, dtype=torch.float64).clone()
    com = torch.as_tensor(center_of_mass, dtype=torch.float64)
    contact = torch.as_tensor(contacts, dtype=torch.bool)
    if contact.ndim == 1:
        contact = contact[:, None]
    if com.ndim != 2 or com.shape != (len(rates) + 1, 3):
        raise ValueError("center_of_mass must have shape [intervals + 1, 3]")
    if contact.ndim != 2 or contact.shape[0] != len(com):
        raise ValueError("contacts must have shape [frames, bodies]")
    if source_dt <= 0 or preparation_duration_s <= 0:
        raise ValueError("source_dt and preparation_duration_s must be positive")
    if velocity_estimation_duration_s <= 0:
        raise ValueError("velocity_estimation_duration_s must be positive")
    if max_phase_rate < 1.0:
        raise ValueError("max_phase_rate must be at least one")
    if minimum_observed_upward_speed_m_s <= 0:
        raise ValueError("minimum observed upward speed must be positive")

    support = contact.any(dim=-1)
    preparation_intervals = max(
        1, int(round(float(preparation_duration_s) / float(source_dt)))
    )
    velocity_intervals = max(
        1, int(round(float(velocity_estimation_duration_s) / float(source_dt)))
    )
    reports = copy.deepcopy(flight_reports)
    for report in reports:
        report["takeoff_phase_propagated"] = False
        start = int(report["start_frame"])
        requested_vz = float(report["initial_vertical_speed_m_s"])
        if (
            not bool(report.get("start_is_support", False))
            or requested_vz <= 0.0
            or start <= 0
            or start >= len(com)
            or not bool(support[start])
        ):
            continue

        support_start = start
        while support_start > 0 and bool(support[support_start - 1]):
            support_start -= 1
        available_intervals = start - support_start
        if available_intervals < 1:
            continue

        estimate_intervals = min(velocity_intervals, available_intervals)
        observed_vz = float(
            (com[start, 2] - com[start - estimate_intervals, 2])
            / (estimate_intervals * float(source_dt))
        )
        report["observed_takeoff_vertical_speed_m_s"] = observed_vz
        if observed_vz < float(minimum_observed_upward_speed_m_s):
            report["takeoff_phase_skip_reason"] = "no_upward_source_extension"
            continue

        first_candidate = max(
            support_start, start - preparation_intervals
        )
        candidate_height = com[first_candidate : start + 1, 2]
        minimum_height = candidate_height.min()
        # Use the last equal minimum: a short flat crouch should end, rather
        # than begin, the launch parameterization.
        minimum_offsets = torch.nonzero(
            torch.isclose(candidate_height, minimum_height), as_tuple=False
        ).flatten()
        launch_start = first_candidate + int(minimum_offsets[-1])
        count = start - launch_start
        displacement = float(com[start, 2] - com[launch_start, 2])
        if count < 1 or displacement <= 1.0e-4:
            report["takeoff_phase_skip_reason"] = "no_positive_extension_distance"
            continue

        acceleration = requested_vz * requested_vz / (2.0 * displacement)
        progress = (
            com[launch_start : start + 1, 2] - com[launch_start, 2]
        ).clamp_min(0.0)
        progress = torch.cummax(progress, dim=0).values
        ideal_time = torch.sqrt(
            2.0 * progress / max(acceleration, 1.0e-8)
        )
        physical_intervals = ideal_time[1:] - ideal_time[:-1]
        desired = torch.where(
            physical_intervals > 1.0e-8,
            float(source_dt) / physical_intervals,
            torch.full_like(physical_intervals, float(max_phase_rate)),
        ).clamp(1.0, float(max_phase_rate))
        rates[launch_start:start] = torch.maximum(
            rates[launch_start:start], desired
        )
        source_terminal_vz = float(
            (com[start, 2] - com[start - 1, 2]) / float(source_dt)
        )
        predicted_terminal_vz = source_terminal_vz * float(desired[-1])
        report.update(
            {
                "takeoff_phase_propagated": True,
                "takeoff_preparation_start_frame": int(launch_start),
                "takeoff_preparation_end_frame": int(start),
                "takeoff_target_phase_rate": float(desired[-1]),
                "predicted_retimed_takeoff_vertical_speed_m_s": float(
                    predicted_terminal_vz
                ),
                "takeoff_constant_acceleration_m_s2": float(acceleration),
                "takeoff_ideal_duration_s": float(ideal_time[-1]),
                "takeoff_interval_phase_rates": [
                    float(value) for value in desired
                ],
            }
        )
    return rates, reports


def build_retiming_grid_from_interval_rates(
    interval_phase_rate: torch.Tensor,
    source_dt: float,
    requested_output_fps: float,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Build a uniform output clock from exact source-interval phase rates."""
    interval_rate = torch.as_tensor(interval_phase_rate, dtype=torch.float64).flatten()
    if interval_rate.numel() < 1 or not torch.isfinite(interval_rate).all():
        raise ValueError("interval_phase_rate must contain finite values")
    if torch.any(interval_rate < 1.0):
        raise ValueError("interval phase rates may not be below one")
    if source_dt <= 0 or requested_output_fps <= 0:
        raise ValueError("source_dt and requested_output_fps must be positive")

    physical_intervals = float(source_dt) / interval_rate
    physical_knots = torch.cat(
        (torch.zeros(1, dtype=torch.float64), torch.cumsum(physical_intervals, dim=0))
    )
    duration = float(physical_knots[-1])
    output_frames = max(2, int(round(duration * requested_output_fps)) + 1)
    output_dt = duration / (output_frames - 1)
    target_time = torch.linspace(0.0, duration, output_frames, dtype=torch.float64)
    upper = torch.searchsorted(physical_knots, target_time, right=True).clamp(
        1, len(physical_knots) - 1
    )
    lower = upper - 1
    span = (physical_knots[upper] - physical_knots[lower]).clamp(min=1e-12)
    alpha = (target_time - physical_knots[lower]) / span
    coordinates = lower.to(torch.float64) + alpha
    coordinates[0] = 0.0
    coordinates[-1] = float(len(interval_rate))
    report = {
        "source_duration_s": float(len(interval_rate) * source_dt),
        "calibrated_duration_s": duration,
        "requested_output_fps": float(requested_output_fps),
        "actual_output_fps": float(1.0 / output_dt),
        "effective_mean_phase_rate": float(
            (len(interval_rate) * source_dt) / duration
        ),
        "phase_rate_min": float(interval_rate.min()),
        "phase_rate_median": float(interval_rate.median()),
        "phase_rate_max": float(interval_rate.max()),
        "source_frames": int(len(interval_rate) + 1),
        "output_frames": int(output_frames),
    }
    return coordinates, output_dt, report


def build_retiming_grid(
    phase_rate: torch.Tensor,
    source_dt: float,
    requested_output_fps: float,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Return fractional source-frame coordinates on a uniform physical clock."""
    rate = torch.as_tensor(phase_rate, dtype=torch.float64).flatten()
    if rate.numel() < 2 or not torch.isfinite(rate).all():
        raise ValueError("phase_rate must contain at least two finite samples")
    if torch.any(rate < 1.0):
        raise ValueError("phase_rate may not be below one")
    if source_dt <= 0 or requested_output_fps <= 0:
        raise ValueError("source_dt and requested_output_fps must be positive")

    interval_rate = 0.5 * (rate[:-1] + rate[1:])
    physical_intervals = float(source_dt) / interval_rate
    physical_knots = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            torch.cumsum(physical_intervals, dim=0),
        )
    )
    duration = float(physical_knots[-1])
    output_frames = max(2, int(round(duration * requested_output_fps)) + 1)
    output_dt = duration / (output_frames - 1)
    target_time = torch.linspace(
        0.0, duration, output_frames, dtype=torch.float64
    )

    # searchsorted plus linear interpolation keeps both endpoints exact and
    # avoids a scipy dependency in training environments.
    upper = torch.searchsorted(physical_knots, target_time, right=True)
    upper = upper.clamp(1, len(physical_knots) - 1)
    lower = upper - 1
    span = (physical_knots[upper] - physical_knots[lower]).clamp(min=1e-12)
    alpha = (target_time - physical_knots[lower]) / span
    source_coordinates = lower.to(torch.float64) + alpha
    source_coordinates[0] = 0.0
    source_coordinates[-1] = float(len(rate) - 1)
    report = {
        "source_duration_s": float((len(rate) - 1) * source_dt),
        "calibrated_duration_s": duration,
        "requested_output_fps": float(requested_output_fps),
        "actual_output_fps": float(1.0 / output_dt),
        "effective_mean_phase_rate": float(
            ((len(rate) - 1) * source_dt) / duration
        ),
        "phase_rate_min": float(rate.min()),
        "phase_rate_median": float(rate.median()),
        "phase_rate_max": float(rate.max()),
        "source_frames": int(len(rate)),
        "output_frames": int(output_frames),
    }
    return source_coordinates, output_dt, report


def _linear_sample(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    source = torch.as_tensor(values)
    coordinates = coordinates.to(dtype=torch.float64, device="cpu")
    lower = torch.floor(coordinates).long().clamp(0, source.shape[0] - 1)
    upper = (lower + 1).clamp(max=source.shape[0] - 1)
    alpha = (coordinates - lower.to(coordinates.dtype)).to(source.dtype)
    alpha = alpha.reshape((-1,) + (1,) * (source.ndim - 1))
    return source[lower] * (1.0 - alpha) + source[upper] * alpha


def _nearest_sample(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    source = torch.as_tensor(values)
    indices = torch.round(coordinates).long().clamp(0, source.shape[0] - 1)
    return source[indices]


def _quaternion_sample(
    quaternions: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    source = torch.as_tensor(quaternions, dtype=torch.float32)
    lower = torch.floor(coordinates).long().clamp(0, source.shape[0] - 1)
    upper = (lower + 1).clamp(max=source.shape[0] - 1)
    alpha = (coordinates - lower.to(coordinates.dtype)).to(source.dtype)
    result = slerp(
        source[lower],
        source[upper],
        alpha[:, None, None],
    )
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp(
        min=1e-8
    )


def load_body_inertial_properties(
    asset_path: Path,
    body_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load simulator-exact body masses and local inertial COM offsets."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(asset_path))
    body_ids = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Body {name!r} is absent from {asset_path}")
        body_ids.append(body_id)
    masses = torch.from_numpy(model.body_mass[body_ids].copy()).to(torch.float32)
    local_inertial_pos = torch.from_numpy(
        model.body_ipos[body_ids].copy()
    ).to(torch.float32)
    if torch.any(masses <= 0) or not torch.isfinite(masses).all():
        raise ValueError("Every articulated body must have a finite positive mass")
    return masses, local_inertial_pos


def compute_system_center_of_mass(
    rigid_body_pos: torch.Tensor,
    rigid_body_rot: torch.Tensor,
    body_masses: torch.Tensor,
    local_inertial_pos: torch.Tensor,
) -> torch.Tensor:
    """Compute the whole-body COM using the same inertials as the simulator."""
    positions = torch.as_tensor(rigid_body_pos)
    rotations = torch.as_tensor(rigid_body_rot)
    masses = torch.as_tensor(body_masses, dtype=positions.dtype, device=positions.device)
    offsets = torch.as_tensor(
        local_inertial_pos, dtype=positions.dtype, device=positions.device
    )
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("rigid_body_pos must have shape [frames,bodies,3]")
    if rotations.shape != positions.shape[:-1] + (4,):
        raise ValueError("rigid_body_rot must have shape [frames,bodies,4]")
    if masses.shape != (positions.shape[1],) or offsets.shape != (
        positions.shape[1],
        3,
    ):
        raise ValueError("mass/inertial arrays must match the rigid-body count")
    world_offsets = torch.einsum(
        "tbij,bj->tbi", quaternion_to_matrix(rotations, w_last=True), offsets
    )
    body_com = positions + world_offsets
    return (body_com * masses[None, :, None]).sum(dim=1) / masses.sum()


def _centered_trajectory_velocity(values: torch.Tensor, dt: float) -> torch.Tensor:
    values = torch.as_tensor(values)
    if values.ndim != 2 or values.shape[-1] != 3 or len(values) < 2:
        raise ValueError("values must have shape [frames>=2,3]")
    if dt <= 0:
        raise ValueError("dt must be positive")
    velocity = torch.empty_like(values)
    velocity[0] = (values[1] - values[0]) / float(dt)
    velocity[-1] = (values[-1] - values[-2]) / float(dt)
    if len(values) > 2:
        velocity[1:-1] = (values[2:] - values[:-2]) / (2.0 * float(dt))
    return velocity


def project_ballistic_root_height(
    root_pos: torch.Tensor,
    contacts: torch.Tensor,
    *,
    dt: float,
    gravity: float = 9.81,
    minimum_flight_duration: float = 0.12,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Project contact-free root-Z spans onto gravity-consistent parabolas.

    Each free span is anchored by its measured endpoint heights.  This changes
    neither XY nor articulated pose and keeps every support boundary exact.  A
    clip containing no contact evidence is intentionally left untouched.
    """
    root = torch.as_tensor(root_pos, dtype=torch.float32).clone()
    contact = torch.as_tensor(contacts, dtype=torch.bool)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError("root_pos must have shape [frames,3]")
    if contact.ndim == 1:
        contact = contact[:, None]
    if contact.ndim != 2 or contact.shape[0] != root.shape[0]:
        raise ValueError("contacts must have shape [frames,bodies]")
    if dt <= 0 or gravity <= 0 or minimum_flight_duration < 0:
        raise ValueError("invalid ballistic projection parameters")
    support = contact.any(dim=-1)
    if not support.any():
        return root, []

    flight = ~support
    padded = torch.cat(
        (torch.zeros(1, dtype=torch.bool), flight.cpu(), torch.zeros(1, dtype=torch.bool))
    )
    transitions = torch.nonzero(padded[1:] != padded[:-1]).flatten().tolist()
    reports: list[dict[str, float]] = []
    for run_start, run_stop in zip(transitions[0::2], transitions[1::2]):
        # run_stop is exclusive. Include the adjacent support sample as an
        # exact boundary when available; clip boundaries serve as one-sided
        # anchors for videos that begin or end in flight.
        anchor_start = run_start - 1 if run_start > 0 else run_start
        anchor_stop = run_stop if run_stop < len(root) else run_stop - 1
        if anchor_stop <= anchor_start:
            continue
        duration = (anchor_stop - anchor_start) * float(dt)
        if duration < minimum_flight_duration:
            continue
        z_start = float(root[anchor_start, 2])
        z_stop = float(root[anchor_stop, 2])
        initial_velocity = (
            z_stop - z_start + 0.5 * float(gravity) * duration * duration
        ) / duration
        times = torch.arange(
            anchor_stop - anchor_start + 1, dtype=root.dtype
        ) * float(dt)
        projected = (
            z_start
            + initial_velocity * times
            - 0.5 * float(gravity) * times.square()
        )
        # Preserve measured anchors bit-for-bit.  Besides avoiding numerical
        # drift in contact heights, this makes the projection idempotent when
        # an already calibrated MotionLib is inspected or repackaged.
        projected[0] = root[anchor_start, 2]
        projected[-1] = root[anchor_stop, 2]
        original = root[anchor_start : anchor_stop + 1, 2].clone()
        root[anchor_start : anchor_stop + 1, 2] = projected
        correction = projected - original
        reports.append(
            {
                "start_frame": int(anchor_start),
                "end_frame": int(anchor_stop),
                "start_is_support": bool(support[anchor_start]),
                "end_is_support": bool(support[anchor_stop]),
                "duration_s": float(duration),
                "initial_vertical_velocity_m_s": float(initial_velocity),
                "final_vertical_velocity_m_s": float(
                    initial_velocity - float(gravity) * duration
                ),
                "correction_abs_max_m": float(correction.abs().max()),
                "correction_rms_m": float(torch.sqrt(correction.square().mean())),
            }
        )
    return root, reports


def project_ballistic_root_translation(
    root_pos: torch.Tensor,
    contacts: torch.Tensor,
    *,
    dt: float,
    gravity: float = 9.81,
    minimum_flight_duration: float = 0.12,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Project a contact-free reference trajectory onto a 3-D ballistic arc.

    In free flight, external force is gravity: horizontal momentum is constant
    and vertical acceleration is ``-g``.  Video root tracks often violate both
    constraints (for example, they accelerate by metres per second while no
    body touches the scene).  Such a target is not merely noisy; it is
    dynamically unreachable by any torque-only controller.

    Each contact-free span is therefore replaced by the unique ballistic arc
    passing through its two measured support/clip anchors.  Articulated pose is
    untouched, and both anchor translations remain bit-exact.  As with the
    vertical-only ablation, a clip with no contact evidence is left unchanged
    rather than inventing an unconstrained global trajectory.
    """
    root = torch.as_tensor(root_pos, dtype=torch.float32).clone()
    contact = torch.as_tensor(contacts, dtype=torch.bool)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError("root_pos must have shape [frames,3]")
    if contact.ndim == 1:
        contact = contact[:, None]
    if contact.ndim != 2 or contact.shape[0] != root.shape[0]:
        raise ValueError("contacts must have shape [frames,bodies]")
    if dt <= 0 or gravity <= 0 or minimum_flight_duration < 0:
        raise ValueError("invalid ballistic projection parameters")
    support = contact.any(dim=-1)
    if not support.any():
        return root, []

    flight = ~support
    padded = torch.cat(
        (torch.zeros(1, dtype=torch.bool), flight.cpu(), torch.zeros(1, dtype=torch.bool))
    )
    transitions = torch.nonzero(padded[1:] != padded[:-1]).flatten().tolist()
    reports: list[dict[str, Any]] = []
    gravity_vector = root.new_tensor((0.0, 0.0, -float(gravity)))
    for run_start, run_stop in zip(transitions[0::2], transitions[1::2]):
        anchor_start = run_start - 1 if run_start > 0 else run_start
        anchor_stop = run_stop if run_stop < len(root) else run_stop - 1
        if anchor_stop <= anchor_start:
            continue
        duration = (anchor_stop - anchor_start) * float(dt)
        if duration < minimum_flight_duration:
            continue

        position_start = root[anchor_start].clone()
        position_stop = root[anchor_stop].clone()
        initial_velocity = (
            position_stop
            - position_start
            - 0.5 * gravity_vector * float(duration * duration)
        ) / float(duration)
        times = torch.arange(
            anchor_stop - anchor_start + 1,
            dtype=root.dtype,
            device=root.device,
        ) * float(dt)
        projected = (
            position_start[None]
            + times[:, None] * initial_velocity[None]
            + 0.5 * times.square()[:, None] * gravity_vector[None]
        )
        projected[0] = position_start
        projected[-1] = position_stop
        original = root[anchor_start : anchor_stop + 1].clone()
        root[anchor_start : anchor_stop + 1] = projected
        correction = projected - original
        final_velocity = initial_velocity + gravity_vector * float(duration)
        reports.append(
            {
                "start_frame": int(anchor_start),
                "end_frame": int(anchor_stop),
                "start_is_support": bool(support[anchor_start]),
                "end_is_support": bool(support[anchor_stop]),
                "duration_s": float(duration),
                "initial_velocity_m_s": [float(v) for v in initial_velocity],
                "final_velocity_m_s": [float(v) for v in final_velocity],
                # Keep scalar fields so existing report readers remain useful.
                "initial_vertical_velocity_m_s": float(initial_velocity[2]),
                "final_vertical_velocity_m_s": float(final_velocity[2]),
                "correction_abs_max_m": float(correction.abs().max()),
                "correction_rms_m": float(torch.sqrt(correction.square().mean())),
                "horizontal_correction_abs_max_m": float(
                    correction[:, :2].abs().max()
                ),
            }
        )
    return root, reports


def enforce_ballistic_com_velocity(
    rigid_body_vel: torch.Tensor,
    root_pos: torch.Tensor,
    system_com: torch.Tensor,
    ballistic_segments: list[dict[str, Any]],
    *,
    dt: float,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Set root reset velocity so the articulated whole-body COM is ballistic.

    Pose changes move the COM relative to the pelvis.  Therefore assigning the
    analytic COM velocity directly to the root is wrong.  We subtract the
    finite-difference velocity of ``COM - root`` and add only the remaining
    common translation to every body's velocity, preserving articulated
    relative motion.
    """
    velocity = torch.as_tensor(rigid_body_vel).clone()
    root = torch.as_tensor(root_pos, dtype=velocity.dtype, device=velocity.device)
    com = torch.as_tensor(system_com, dtype=velocity.dtype, device=velocity.device)
    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("rigid_body_vel must have shape [frames,bodies,3]")
    if root.shape != (len(velocity), 3) or com.shape != root.shape:
        raise ValueError("root_pos and system_com must have shape [frames,3]")
    relative_com_velocity = _centered_trajectory_velocity(com - root, dt)

    for segment in ballistic_segments:
        if "initial_velocity_m_s" not in segment:
            raise ValueError("COM ballistic segment requires XYZ initial velocity")
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start < 0 or end < start or end >= len(velocity):
            raise ValueError("ballistic segment lies outside the motion")
        stop = end if bool(segment.get("end_is_support", False)) else end + 1
        if stop <= start:
            continue
        frame_ids = torch.arange(start, stop, device=velocity.device)
        elapsed = (frame_ids - start).to(velocity.dtype) * float(dt)
        desired_com_velocity = velocity.new_tensor(
            segment["initial_velocity_m_s"]
        )[None].expand(len(frame_ids), -1).clone()
        desired_com_velocity[:, 2] -= float(gravity) * elapsed
        desired_root_velocity = (
            desired_com_velocity - relative_com_velocity[frame_ids]
        )
        delta = desired_root_velocity - velocity[frame_ids, 0]
        velocity[frame_ids] += delta[:, None, :]
    return velocity


def enforce_ballistic_translation_velocity(
    rigid_body_vel: torch.Tensor,
    ballistic_segments: list[dict[str, float]],
    *,
    dt: float,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Restore the analytic root translation velocity on ballistic spans.

    The generic FK velocity estimator deliberately takes the minimum magnitude
    across several finite-difference horizons.  That is useful for noisy video
    poses, but it biases a real parabolic trajectory toward zero, especially at
    a clip boundary.  A simulator reset then starts with too little take-off
    velocity and immediately falls below the reference.

    For spans already certified by support/contact anchors, replace only the
    common vertical translation component with ``v0 - g*t``.  Adding the same
    delta to every body preserves all articulation-induced relative velocity.
    A terminal support sample is excluded: it represents the post-impact state,
    not the incoming ballistic velocity.  No-contact clips have no segments and
    are returned unchanged.
    """
    velocity = torch.as_tensor(rigid_body_vel).clone()
    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("rigid_body_vel must have shape [frames,bodies,3]")
    if dt <= 0 or gravity <= 0:
        raise ValueError("dt and gravity must be positive")

    frame_count = velocity.shape[0]
    for segment in ballistic_segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start < 0 or end < start or end >= frame_count:
            raise ValueError("ballistic segment lies outside rigid_body_vel")
        # At a support endpoint the collision impulse has already happened, so
        # retain the FK/contact velocity for that sample.  A clip endpoint still
        # in flight receives the analytic velocity.
        stop = end if bool(segment.get("end_is_support", False)) else end + 1
        if stop <= start:
            continue
        frame_ids = torch.arange(start, stop, device=velocity.device)
        elapsed = (frame_ids - start).to(velocity.dtype) * float(dt)
        if "initial_velocity_m_s" in segment:
            initial = velocity.new_tensor(segment["initial_velocity_m_s"])
            if initial.shape != (3,):
                raise ValueError("initial_velocity_m_s must contain XYZ")
            desired_root_velocity = initial[None].expand(len(frame_ids), -1).clone()
            desired_root_velocity[:, 2] -= float(gravity) * elapsed
            delta = desired_root_velocity - velocity[frame_ids, 0]
            velocity[frame_ids] += delta[:, None, :]
        else:
            desired_root_vz = (
                float(segment["initial_vertical_velocity_m_s"])
                - float(gravity) * elapsed
            )
            root_vz = velocity[frame_ids, 0, 2]
            delta = desired_root_vz - root_vz
            velocity[frame_ids, :, 2] += delta[:, None]
    return velocity


def _resolve_asset(asset: str, repository_root: Path) -> Path:
    path = Path(asset).expanduser()
    candidates = (path, repository_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Motion asset does not exist: {asset}")


def retime_motion(
    motion: dict[str, Any],
    *,
    repository_root: Path,
    requested_output_fps: float = 30.0,
    unreliable_below: float = 0.25,
    reliable_above: float = 0.45,
    max_phase_rate: float = 3.0,
    transition_radius: int = 2,
    project_ballistic_flight: bool = False,
    project_ballistic_translation: bool = False,
    gravity: float = 9.81,
    minimum_flight_duration: float = 0.12,
    recover_latent_foot_support: bool = False,
    latent_foot_body_ids: tuple[int, ...] = (3, 4, 7, 8),
    latent_support_maximum_clearance_m: float = 0.42,
    latent_support_maximum_speed_m_s: float = 1.8,
    reachability_aware_phase: bool = False,
    propagate_takeoff_phase: bool = False,
    takeoff_preparation_duration_s: float = 0.40,
    takeoff_velocity_estimation_duration_s: float = 0.14,
    takeoff_max_phase_rate: float = 4.0,
    promote_reachable_plan_reliability: bool = False,
    reachable_plan_reliability_floor: float = 0.85,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        "gts",
        "lrs",
        "contacts",
        "motion_dt",
        "motion_num_frames",
        "motion_asset_files",
        "root_physical_reliability",
    }
    missing = required.difference(motion)
    if missing:
        raise KeyError(f"MotionLib is missing fields: {sorted(missing)}")

    frame_counts = torch.as_tensor(motion["motion_num_frames"], dtype=torch.long)
    if int(frame_counts.sum()) != len(motion["gts"]):
        raise ValueError("motion_num_frames does not match packed frame count")
    if len(frame_counts) != len(motion["motion_asset_files"]):
        raise ValueError("motion_asset_files does not match number of motions")
    if not 0.0 <= reachable_plan_reliability_floor <= 1.0:
        raise ValueError("reachable_plan_reliability_floor must lie in [0, 1]")
    if reachability_aware_phase and not project_ballistic_translation:
        raise ValueError(
            "reachability-aware phase requires --project-ballistic-translation "
            "so the selected clock and root dynamics remain consistent"
        )
    if propagate_takeoff_phase and not reachability_aware_phase:
        raise ValueError(
            "takeoff phase propagation requires --reachability-aware-phase"
        )

    frame_outputs: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "gts",
            "grs",
            "gvs",
            "gavs",
            "dvs",
            "dps",
            "contacts",
            "lrs",
            "root_reference_reliability",
            "root_observation_reliability",
            "root_physical_reliability",
            "root_translation_correction",
            "source_frame_coordinate",
            "source_phase_rate",
            "root_ballistic_correction",
            "root_plan_reliability",
            "root_observation_anchored_reliability",
            "latent_support_recovered",
        )
    }
    new_counts: list[int] = []
    new_dts: list[float] = []
    reports: list[dict[str, Any]] = []
    start = 0
    for motion_id, count_tensor in enumerate(frame_counts):
        count = int(count_tensor)
        stop = start + count
        source_dt = float(torch.as_tensor(motion["motion_dt"])[motion_id])
        asset_path = _resolve_asset(
            motion["motion_asset_files"][motion_id], repository_root
        )
        kinematic_info = extract_kinematic_info(str(asset_path)).to(
            torch.device("cpu"), torch.float32
        )
        reliability = torch.as_tensor(
            motion["root_physical_reliability"][start:stop]
        )
        source_contacts_original = torch.as_tensor(
            motion["contacts"][start:stop], dtype=torch.bool
        )
        source_contacts = source_contacts_original.clone()
        source_positions = torch.as_tensor(
            motion["gts"][start:stop], dtype=torch.float32
        ).clone()
        source_root_correction = torch.zeros(count, 3, dtype=torch.float32)
        if "root_translation_correction" in motion:
            source_root_correction.copy_(
                torch.as_tensor(
                    motion["root_translation_correction"][start:stop],
                    dtype=torch.float32,
                )
            )
        latent_diagnostics: dict[str, torch.Tensor] | None = None
        support_correction = torch.zeros(count, dtype=torch.float32)
        if recover_latent_foot_support:
            for key in ("surface_contact_clearance", "surface_contact_speed"):
                if key not in motion:
                    raise KeyError(
                        f"--recover-latent-foot-support requires {key!r}"
                    )
            source_clearance = torch.as_tensor(
                motion["surface_contact_clearance"][start:stop]
            )
            source_speed = torch.as_tensor(
                motion["surface_contact_speed"][start:stop]
            )
            source_contacts, latent_diagnostics = (
                recover_seed_connected_foot_support(
                    source_contacts_original,
                    source_clearance,
                    source_speed,
                    latent_foot_body_ids,
                    maximum_clearance_m=latent_support_maximum_clearance_m,
                    maximum_speed_m_s=latent_support_maximum_speed_m_s,
                )
            )
            observation_reliability = motion.get(
                "root_observation_reliability",
                motion.get("root_reference_reliability"),
            )
            support_correction = solve_surface_support_root_correction(
                source_contacts,
                source_contacts_original,
                source_clearance,
                observation_reliability=(
                    None
                    if observation_reliability is None
                    else torch.as_tensor(observation_reliability)[start:stop]
                ),
            )
            source_positions[..., 2] += support_correction[:, None]
            source_root_correction[:, 2] += support_correction

        body_masses: torch.Tensor | None = None
        local_inertial_pos: torch.Tensor | None = None
        reachability_reports: list[dict[str, Any]] = []
        if reachability_aware_phase:
            body_masses, local_inertial_pos = load_body_inertial_properties(
                asset_path, kinematic_info.body_names
            )
            source_com = compute_system_center_of_mass(
                source_positions,
                torch.as_tensor(motion["grs"][start:stop], dtype=torch.float32),
                body_masses,
                local_inertial_pos,
            )
            interval_rate, reachability_reports = (
                reachability_aware_interval_phase_rates(
                    source_com,
                    source_contacts,
                    source_dt=source_dt,
                    gravity=gravity,
                    max_phase_rate=max_phase_rate,
                )
            )
            if propagate_takeoff_phase:
                interval_rate, reachability_reports = (
                    propagate_ballistic_takeoff_phase_rates(
                        interval_rate,
                        source_com,
                        source_contacts,
                        reachability_reports,
                        source_dt=source_dt,
                        preparation_duration_s=(
                            takeoff_preparation_duration_s
                        ),
                        velocity_estimation_duration_s=(
                            takeoff_velocity_estimation_duration_s
                        ),
                        max_phase_rate=takeoff_max_phase_rate,
                    )
                )
            coordinates, output_dt, report = (
                build_retiming_grid_from_interval_rates(
                    interval_rate, source_dt, requested_output_fps
                )
            )
            rate = torch.empty(count, dtype=torch.float64)
            rate[0] = interval_rate[0]
            rate[-1] = interval_rate[-1]
            if count > 2:
                rate[1:-1] = 0.5 * (interval_rate[:-1] + interval_rate[1:])
        else:
            rate = phase_rate_from_reliability(
                reliability,
                source_contacts,
                unreliable_below=unreliable_below,
                reliable_above=reliable_above,
                max_phase_rate=max_phase_rate,
                transition_radius=transition_radius,
            )
            coordinates, output_dt, report = build_retiming_grid(
                rate, source_dt, requested_output_fps
            )

        root_pos = _linear_sample(
            source_positions[:, 0], coordinates
        ).float()
        sampled_contacts = _nearest_sample(source_contacts, coordinates).bool()
        root_before_ballistic = root_pos.clone()
        ballistic_segments: list[dict[str, Any]] = []
        if project_ballistic_flight:
            root_pos, ballistic_segments = project_ballistic_root_height(
                root_pos,
                sampled_contacts,
                dt=output_dt,
                gravity=gravity,
                minimum_flight_duration=minimum_flight_duration,
            )
        local_quat = _quaternion_sample(
            torch.as_tensor(motion["lrs"])[start:stop], coordinates
        ).float()
        joint_rot_mats = quaternion_to_matrix(local_quat, w_last=True)
        actual_fps = 1.0 / output_dt
        if project_ballistic_translation:
            # Ballistic dynamics constrain the whole-body mass centre, not the
            # pelvis.  First evaluate the retimed articulation, project its COM,
            # then apply that one shared correction to the root translation.
            preliminary_state = fk_from_transforms_with_velocities(
                kinematic_info=kinematic_info,
                root_pos=root_pos,
                joint_rot_mats=joint_rot_mats,
                fps=actual_fps,
                compute_velocities=False,
            )
            if body_masses is None or local_inertial_pos is None:
                body_masses, local_inertial_pos = load_body_inertial_properties(
                    asset_path, kinematic_info.body_names
                )
            original_com = compute_system_center_of_mass(
                preliminary_state.rigid_body_pos,
                preliminary_state.rigid_body_rot,
                body_masses,
                local_inertial_pos,
            )
            projected_com, ballistic_segments = project_ballistic_root_translation(
                original_com,
                sampled_contacts,
                dt=output_dt,
                gravity=gravity,
                minimum_flight_duration=minimum_flight_duration,
            )
            root_pos = root_pos + (projected_com - original_com)
        state = fk_from_transforms_with_velocities(
            kinematic_info=kinematic_info,
            root_pos=root_pos,
            joint_rot_mats=joint_rot_mats,
            fps=actual_fps,
            compute_velocities=True,
            velocity_max_horizon=3,
        )
        if project_ballistic_translation and ballistic_segments:
            assert body_masses is not None and local_inertial_pos is not None
            corrected_com = compute_system_center_of_mass(
                state.rigid_body_pos,
                state.rigid_body_rot,
                body_masses,
                local_inertial_pos,
            )
            state.rigid_body_vel = enforce_ballistic_com_velocity(
                state.rigid_body_vel,
                root_pos,
                corrected_com,
                ballistic_segments,
                dt=output_dt,
                gravity=gravity,
            )
        elif ballistic_segments:
            state.rigid_body_vel = enforce_ballistic_translation_velocity(
                state.rigid_body_vel,
                ballistic_segments,
                dt=output_dt,
                gravity=gravity,
            )
        qpos = extract_qpos_from_transforms(
            kinematic_info=kinematic_info,
            root_pos=root_pos,
            joint_rot_mats=joint_rot_mats,
            multi_dof_decomposition_method="exp_map",
        )

        frame_outputs["gts"].append(state.rigid_body_pos.float())
        frame_outputs["grs"].append(state.rigid_body_rot.float())
        frame_outputs["gvs"].append(state.rigid_body_vel.float())
        frame_outputs["gavs"].append(state.rigid_body_ang_vel.float())
        frame_outputs["dps"].append(qpos[:, 7:].float())
        frame_outputs["dvs"].append(
            compute_angular_velocity(
                joint_rot_mats[:, 1:], fps=actual_fps
            ).reshape(len(coordinates), -1).float()
        )
        frame_outputs["contacts"].append(
            sampled_contacts
        )
        frame_outputs["lrs"].append(local_quat)
        source_plan_reliability = torch.ones(count, dtype=torch.float32)
        if reachability_reports:
            for flight_report in reachability_reports:
                if not flight_report["within_speed_limits"]:
                    source_plan_reliability[
                        int(flight_report["start_frame"]) : int(
                            flight_report["end_frame"]
                        )
                        + 1
                    ] = 0.10
        elif not reachability_aware_phase:
            source_plan_reliability = reliability.to(torch.float32).clamp(0.0, 1.0)
        sampled_plan_reliability = _linear_sample(
            source_plan_reliability, coordinates
        ).to(torch.float32)
        frame_outputs["root_plan_reliability"].append(
            sampled_plan_reliability
        )
        if latent_diagnostics is not None:
            frame_outputs["latent_support_recovered"].append(
                _nearest_sample(
                    latent_diagnostics["recovered"], coordinates
                ).bool()
            )
        for key in (
            "root_reference_reliability",
            "root_observation_reliability",
            "root_physical_reliability",
        ):
            if key in motion:
                sampled = _linear_sample(
                    torch.as_tensor(motion[key])[start:stop], coordinates
                ).to(torch.float32)
                if key == "root_reference_reliability" and promote_reachable_plan_reliability:
                    frame_outputs["root_observation_anchored_reliability"].append(
                        sampled.clone()
                    )
                    sampled = torch.maximum(
                        sampled,
                        float(reachable_plan_reliability_floor)
                        * sampled_plan_reliability,
                    )
                frame_outputs[key].append(sampled)
        frame_outputs["root_translation_correction"].append(
            _linear_sample(source_root_correction, coordinates).to(torch.float32)
        )
        frame_outputs["source_frame_coordinate"].append(coordinates.float())
        frame_outputs["source_phase_rate"].append(
            _linear_sample(rate.float(), coordinates).float()
        )
        frame_outputs["root_ballistic_correction"].append(
            (root_pos - root_before_ballistic).float()
        )
        report.update(
            {
                "motion_id": int(motion_id),
                "source_dt": source_dt,
                "output_dt": output_dt,
                "contact_locked_frames": int(source_contacts.any(dim=-1).sum()),
                "original_contact_frames": int(
                    source_contacts_original.any(dim=-1).sum()
                ),
                "latent_support_recovered_frames": int(
                    0
                    if latent_diagnostics is None
                    else latent_diagnostics["recovered"].sum()
                ),
                "support_root_correction_min_m": float(
                    support_correction.min()
                ),
                "support_root_correction_max_m": float(
                    support_correction.max()
                ),
                "reachability_flight_segments": reachability_reports,
                "ballistic_segment_count": int(len(ballistic_segments)),
                "ballistic_segments": ballistic_segments,
            }
        )
        reports.append(report)
        new_counts.append(len(coordinates))
        new_dts.append(output_dt)
        start = stop

    output = copy.deepcopy(motion)
    for key, chunks in frame_outputs.items():
        if chunks:
            output[key] = torch.cat(chunks, dim=0)
        elif key in output:
            del output[key]
    # These source-time diagnostics are no longer valid after support recovery
    # and non-uniform retiming.  The final contacts and calibration report are
    # authoritative; retaining stale arrays would invite accidental reuse.
    if recover_latent_foot_support or reachability_aware_phase:
        output.pop("surface_contact_clearance", None)
        output.pop("surface_contact_speed", None)
    if recover_latent_foot_support:
        source_method = str(output.get("contact_label_method", "surface"))
        output["contact_label_method"] = (
            source_method + "+seed_connected_latent_foot_v1"
        )
    counts_tensor = torch.tensor(new_counts, dtype=torch.long)
    starts = counts_tensor.roll(1)
    starts[0] = 0
    output["length_starts"] = starts.cumsum(0)
    output["motion_num_frames"] = counts_tensor
    output["motion_dt"] = torch.tensor(new_dts, dtype=torch.float32)
    output["motion_lengths"] = torch.tensor(
        [(count - 1) * dt for count, dt in zip(new_counts, new_dts)],
        dtype=torch.float32,
    )
    output["motion_files"] = tuple(
        f"{name}_physical_phase_calibrated"
        for name in motion.get(
            "motion_files", tuple(f"motion_{i}" for i in range(len(new_counts)))
        )
    )
    calibration = {
        "method": (
            "latent_support_reachability_takeoff_com_projection_v2"
            if (
                recover_latent_foot_support
                and reachability_aware_phase
                and propagate_takeoff_phase
                and project_ballistic_translation
            )
            else "latent_support_reachability_com_projection_v1"
            if (
                recover_latent_foot_support
                and reachability_aware_phase
                and project_ballistic_translation
            )
            else "support_anchored_whole_body_com_ballistic_phase_projection_v3"
            if project_ballistic_translation
            else (
                "support_anchored_ballistic_phase_projection_v1"
                if project_ballistic_flight
                else "physical_reliability_phase_calibration_v1"
            )
        ),
        "unreliable_below": float(unreliable_below),
        "reliable_above": float(reliable_above),
        "max_phase_rate": float(max_phase_rate),
        "transition_radius": int(transition_radius),
        "requested_output_fps": float(requested_output_fps),
        "project_ballistic_flight": bool(project_ballistic_flight),
        "project_ballistic_translation": bool(project_ballistic_translation),
        "gravity_m_s2": float(gravity),
        "minimum_flight_duration_s": float(minimum_flight_duration),
        "recover_latent_foot_support": bool(recover_latent_foot_support),
        "latent_foot_body_ids": [int(v) for v in latent_foot_body_ids],
        "latent_support_maximum_clearance_m": float(
            latent_support_maximum_clearance_m
        ),
        "latent_support_maximum_speed_m_s": float(
            latent_support_maximum_speed_m_s
        ),
        "reachability_aware_phase": bool(reachability_aware_phase),
        "propagate_takeoff_phase": bool(propagate_takeoff_phase),
        "takeoff_preparation_duration_s": float(
            takeoff_preparation_duration_s
        ),
        "takeoff_velocity_estimation_duration_s": float(
            takeoff_velocity_estimation_duration_s
        ),
        "takeoff_max_phase_rate": float(takeoff_max_phase_rate),
        "promote_reachable_plan_reliability": bool(
            promote_reachable_plan_reliability
        ),
        "reachable_plan_reliability_floor": float(
            reachable_plan_reliability_floor
        ),
        "reports": reports,
    }
    output["physical_phase_calibration"] = calibration
    if isinstance(output.get("root_projection_metadata"), dict):
        output["root_projection_metadata"]["physical_phase_calibration"] = calibration
    return output, reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--unreliable-below", type=float, default=0.25)
    parser.add_argument("--reliable-above", type=float, default=0.45)
    parser.add_argument("--max-phase-rate", type=float, default=3.0)
    parser.add_argument("--transition-radius", type=int, default=2)
    parser.add_argument("--project-ballistic-flight", action="store_true")
    parser.add_argument(
        "--project-ballistic-translation",
        action="store_true",
        help="Project free-flight root XYZ onto constant-horizontal-momentum ballistic arcs.",
    )
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--minimum-flight-duration", type=float, default=0.12)
    parser.add_argument(
        "--recover-latent-foot-support",
        action="store_true",
        help=(
            "Extend observed foot-contact runs through plausible occluded "
            "surface candidates and project the shared root gauge onto them."
        ),
    )
    parser.add_argument(
        "--latent-foot-body-ids",
        type=str,
        default="3,4,7,8",
        help="Comma-separated MotionLib body indices for ankles/toes.",
    )
    parser.add_argument(
        "--latent-support-maximum-clearance",
        type=float,
        default=0.42,
    )
    parser.add_argument(
        "--latent-support-maximum-speed",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--reachability-aware-phase",
        action="store_true",
        help=(
            "Choose each contact-free duration from ballistic endpoint "
            "reachability instead of confidence alone."
        ),
    )
    parser.add_argument(
        "--propagate-takeoff-phase",
        action="store_true",
        help=(
            "Smoothly propagate each ballistic launch rate into the preceding "
            "support articulation so takeoff velocity is continuous."
        ),
    )
    parser.add_argument(
        "--takeoff-preparation-duration",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--takeoff-velocity-estimation-duration",
        type=float,
        default=0.14,
    )
    parser.add_argument(
        "--takeoff-max-phase-rate",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--promote-reachable-plan-reliability",
        action="store_true",
        help=(
            "Expose the validated physical plan as a strong controller target "
            "while retaining the original confidence in a separate field."
        ),
    )
    parser.add_argument(
        "--reachable-plan-reliability-floor",
        type=float,
        default=0.85,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.project_ballistic_flight and args.project_ballistic_translation:
        raise ValueError(
            "Choose only one of --project-ballistic-flight and "
            "--project-ballistic-translation"
        )
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists (pass --force): {output_path}")
    motion = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(motion, dict):
        raise TypeError("Expected a packaged MotionLib dictionary")
    repository_root = Path(__file__).resolve().parents[2]
    latent_foot_body_ids = tuple(
        int(value.strip())
        for value in args.latent_foot_body_ids.split(",")
        if value.strip()
    )
    if not latent_foot_body_ids:
        raise ValueError("--latent-foot-body-ids must contain at least one index")
    output, reports = retime_motion(
        motion,
        repository_root=repository_root,
        requested_output_fps=args.output_fps,
        unreliable_below=args.unreliable_below,
        reliable_above=args.reliable_above,
        max_phase_rate=args.max_phase_rate,
        transition_radius=args.transition_radius,
        project_ballistic_flight=args.project_ballistic_flight,
        project_ballistic_translation=args.project_ballistic_translation,
        gravity=args.gravity,
        minimum_flight_duration=args.minimum_flight_duration,
        recover_latent_foot_support=args.recover_latent_foot_support,
        latent_foot_body_ids=latent_foot_body_ids,
        latent_support_maximum_clearance_m=(
            args.latent_support_maximum_clearance
        ),
        latent_support_maximum_speed_m_s=args.latent_support_maximum_speed,
        reachability_aware_phase=args.reachability_aware_phase,
        propagate_takeoff_phase=args.propagate_takeoff_phase,
        takeoff_preparation_duration_s=args.takeoff_preparation_duration,
        takeoff_velocity_estimation_duration_s=(
            args.takeoff_velocity_estimation_duration
        ),
        takeoff_max_phase_rate=args.takeoff_max_phase_rate,
        promote_reachable_plan_reliability=(
            args.promote_reachable_plan_reliability
        ),
        reachable_plan_reliability_floor=(
            args.reachable_plan_reliability_floor
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report_path = args.report
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(output["physical_phase_calibration"], indent=2),
            encoding="utf-8",
        )
    print(f"Saved physically phase-calibrated MotionLib: {output_path}")
    for report in reports:
        print(
            "motion={motion_id}: {source_frames} frames/{source_duration_s:.3f}s "
            "-> {output_frames} frames/{calibrated_duration_s:.3f}s, "
            "mean phase rate={effective_mean_phase_rate:.3f}x, "
            "range={phase_rate_min:.3f}-{phase_rate_max:.3f}x".format(**report)
        )


if __name__ == "__main__":
    main()
