# SPDX-License-Identifier: Apache-2.0
"""Denoise non-root local joint rotations in a packaged MotionLib.

The filter is a zero-phase normalized quaternion convolution.  It preserves
the root trajectory and root orientation exactly, then recomputes FK, DOF
coordinates and velocity targets using each motion's own MJCF body shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protomotions.components.pose_lib import (
    compute_angular_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import quaternion_to_matrix

from data.scripts.robustify_motion_velocities import robustify_motion_velocities


def _binomial_kernel(window: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")
    order = window - 1
    coeffs = [1]
    for _ in range(order):
        coeffs = [1] + [a + b for a, b in zip(coeffs[:-1], coeffs[1:])] + [1]
    kernel = torch.tensor(coeffs, dtype=dtype, device=device)
    return kernel / kernel.sum()


def smooth_quaternions(quaternions: torch.Tensor, window: int) -> torch.Tensor:
    """Sign-aligned chordal quaternion average over time."""
    if quaternions.ndim != 3 or quaternions.shape[-1] != 4:
        raise ValueError("Expected quaternions with shape [frames, bodies, 4].")
    kernel = _binomial_kernel(window, quaternions.dtype, quaternions.device)
    radius = window // 2
    frames = quaternions.shape[0]
    frame_ids = torch.arange(frames, device=quaternions.device)
    samples = []
    center = quaternions
    for offset in range(-radius, radius + 1):
        ids = (frame_ids + offset).clamp(0, frames - 1)
        candidate = quaternions[ids]
        candidate = torch.where(
            (candidate * center).sum(dim=-1, keepdim=True) < 0,
            -candidate,
            candidate,
        )
        samples.append(candidate)
    stacked = torch.stack(samples, dim=0)
    averaged = (stacked * kernel[:, None, None, None]).sum(dim=0)
    return averaged / averaged.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def denoise_motion(motion: dict, window: int, velocity_window: int) -> tuple[dict, dict]:
    required = {"gts", "grs", "lrs", "motion_num_frames", "motion_dt", "motion_asset_files"}
    missing = required.difference(motion)
    if missing:
        raise KeyError(f"MotionLib is missing keys: {sorted(missing)}")

    output = dict(motion)
    counts = torch.as_tensor(motion["motion_num_frames"], dtype=torch.long).tolist()
    dts = torch.as_tensor(motion["motion_dt"], dtype=torch.float64).tolist()
    assets = list(motion["motion_asset_files"])
    if not (len(counts) == len(dts) == len(assets)):
        raise ValueError("Motion counts, dt values, and shape assets must align.")

    fields = {key: [] for key in ("gts", "grs", "gvs", "gavs", "dps", "dvs", "lrs")}
    report = {"window": window, "velocity_window": velocity_window, "motions": []}
    start = 0
    for motion_id, (count, dt, asset) in enumerate(zip(counts, dts, assets)):
        stop = start + count
        source_lrs = torch.as_tensor(motion["lrs"])[start:stop].float()
        filtered_lrs = source_lrs.clone()
        # Body 0 is the world/root rotation and must remain exact.
        filtered_lrs[:, 1:] = smooth_quaternions(source_lrs[:, 1:], window)
        root_pos = torch.as_tensor(motion["gts"])[start:stop, 0].float()
        kinematic_info = extract_kinematic_info(str(asset)).to(torch.device("cpu"), torch.float32)
        joint_rot_mats = quaternion_to_matrix(filtered_lrs, w_last=True)
        fps = float(1.0 / dt)
        state = fk_from_transforms_with_velocities(
            kinematic_info=kinematic_info,
            root_pos=root_pos,
            joint_rot_mats=joint_rot_mats,
            fps=fps,
            compute_velocities=True,
            velocity_max_horizon=3,
        )
        qpos = extract_qpos_from_transforms(
            kinematic_info=kinematic_info,
            root_pos=root_pos,
            joint_rot_mats=joint_rot_mats,
            multi_dof_decomposition_method="exp_map",
        )
        dof_pos = qpos[:, 7:]
        dof_vel = compute_angular_velocity(
            batched_robot_rot_mats=joint_rot_mats[:, 1:], fps=fps
        ).reshape(count, -1)
        fields["gts"].append(state.rigid_body_pos)
        fields["grs"].append(state.rigid_body_rot)
        fields["gvs"].append(state.rigid_body_vel)
        fields["gavs"].append(state.rigid_body_ang_vel)
        fields["dps"].append(dof_pos)
        fields["dvs"].append(dof_vel)
        fields["lrs"].append(filtered_lrs)

        dot = (source_lrs[:, 1:] * filtered_lrs[:, 1:]).sum(dim=-1).abs().clamp(max=1.0)
        angle_deg = torch.rad2deg(2.0 * torch.acos(dot))
        report["motions"].append({
            "motion_id": motion_id,
            "frames": count,
            "joint_rotation_change_deg_mean": float(angle_deg.mean()),
            "joint_rotation_change_deg_p95": float(torch.quantile(angle_deg, 0.95)),
            "root_position_max_abs_change": float(
                (state.rigid_body_pos[:, 0] - root_pos).abs().max()
            ),
            "root_rotation_max_abs_change": float(
                (filtered_lrs[:, 0] - source_lrs[:, 0]).abs().max()
            ),
        })
        start = stop

    for key, chunks in fields.items():
        output[key] = torch.cat(chunks, dim=0).to(dtype=torch.as_tensor(motion[key]).dtype)

    if velocity_window:
        output, velocity_report = robustify_motion_velocities(
            output, window_length=velocity_window, polyorder=2
        )
        report["velocity_report"] = velocity_report
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--velocity-window", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists (pass --force): {args.output}")
    motion = torch.load(args.input, map_location="cpu", weights_only=False)
    output, report = denoise_motion(motion, args.window, args.velocity_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved pose-denoised MotionLib: {args.output}")
    print(report)


if __name__ == "__main__":
    main()
