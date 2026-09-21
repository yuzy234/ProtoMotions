# SPDX-License-Identifier: Apache-2.0
"""Denoise MotionLib velocity targets without changing the reference poses.

Video-derived SMPL positions and rotations can be acceptable while their finite-
difference velocity fields are very noisy.  Those fields are used both at reset
and by mimic observations/rewards.  This utility creates an explicit ablation
that smooths only ``gvs``, ``gavs``, and ``dvs``; positions, rotations, DOF poses,
contacts, body shape, timing, and world trajectory remain byte-for-byte equal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter


VELOCITY_KEYS = ("gvs", "gavs", "dvs")
PRESERVED_KEYS = ("gts", "grs", "dps", "lrs", "contacts")


def _valid_window(requested: int, num_frames: int, polyorder: int) -> int:
    window = min(requested, num_frames if num_frames % 2 else num_frames - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        raise ValueError(
            f"Motion is too short for window={requested}, polyorder={polyorder}."
        )
    return window


def robustify_motion_velocities(
    motion: dict,
    window_length: int = 7,
    polyorder: int = 2,
) -> tuple[dict, dict[str, dict[str, float]]]:
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer >= 3.")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder must satisfy 0 <= polyorder < window_length.")
    missing = set(VELOCITY_KEYS + PRESERVED_KEYS).difference(motion)
    if missing:
        raise KeyError(f"MotionLib is missing keys: {sorted(missing)}")

    output = dict(motion)
    report: dict[str, dict[str, float]] = {}
    num_frames = int(torch.as_tensor(motion["gts"]).shape[0])
    frame_counts = torch.as_tensor(
        motion.get("motion_num_frames", [num_frames]), dtype=torch.long
    ).tolist()
    if sum(frame_counts) != num_frames:
        raise ValueError(
            "motion_num_frames does not sum to the number of packed frames: "
            f"{sum(frame_counts)} != {num_frames}."
        )

    for key in VELOCITY_KEYS:
        source_tensor = torch.as_tensor(motion[key])
        source = source_tensor.detach().cpu().numpy()
        # A packaged MotionLib may contain several clips. Filter each clip
        # independently so the last velocity of one person/motion never leaks
        # into the first velocity of the next one.
        filtered_chunks = []
        start = 0
        for count in frame_counts:
            chunk = source[start : start + count]
            if count <= polyorder:
                # Keep very short clips usable in the default pipeline. There
                # are not enough samples to fit the requested polynomial.
                filtered_chunks.append(chunk.copy())
            else:
                window = _valid_window(window_length, count, polyorder)
                filtered_chunks.append(
                    savgol_filter(
                        chunk,
                        window_length=window,
                        polyorder=polyorder,
                        axis=0,
                        mode="interp",
                    ).astype(source.dtype, copy=False)
                )
            start += count
        filtered = np.concatenate(filtered_chunks, axis=0)
        output[key] = torch.as_tensor(filtered).to(
            dtype=source_tensor.dtype,
            device=source_tensor.device,
        )

        delta_before_chunks = []
        delta_after_chunks = []
        start = 0
        for count in frame_counts:
            delta_before_chunks.append(np.diff(source[start : start + count], axis=0))
            delta_after_chunks.append(np.diff(filtered[start : start + count], axis=0))
            start += count
        delta_before = np.concatenate(delta_before_chunks, axis=0)
        delta_after = np.concatenate(delta_after_chunks, axis=0)
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
        report[key] = {
            "value_rms_before": rms(source),
            "value_rms_after": rms(filtered),
            "removed_rms": rms(source - filtered),
            "frame_delta_rms_before": rms(delta_before),
            "frame_delta_rms_after": rms(delta_after),
        }

    # Guard against accidentally changing the kinematic/reference semantics.
    for key in PRESERVED_KEYS:
        if torch.is_tensor(motion[key]) and output[key].data_ptr() != motion[key].data_ptr():
            raise RuntimeError(f"Preserved field {key} was unexpectedly copied or changed.")
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=7)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists (pass --force): {output_path}")

    motion = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(motion, dict):
        raise TypeError("Expected a packaged MotionLib dictionary.")
    output, report = robustify_motion_velocities(
        motion,
        window_length=args.window_length,
        polyorder=args.polyorder,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"Saved velocity-robust MotionLib: {output_path}")
    print(f"Pose/trajectory/contact fields unchanged: {', '.join(PRESERVED_KEYS)}")
    for key, metrics in report.items():
        print(
            f"{key}: frame_delta_rms "
            f"{metrics['frame_delta_rms_before']:.6f} -> "
            f"{metrics['frame_delta_rms_after']:.6f}; "
            f"removed_rms={metrics['removed_rms']:.6f}"
        )


if __name__ == "__main__":
    main()
