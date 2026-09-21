# SPDX-License-Identifier: Apache-2.0
"""Compare simulated MotionLib trajectories against one noisy reference clip.

The metrics deliberately separate tracking fidelity from physical/natural motion
quality.  A policy can obtain a high frame-wise reward while still oscillating,
twisting joints, sliding its feet, or repeatedly depenetrating from the terrain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from protomotions.components.terrains.mesh_terrain import (
    query_height_layers_bilinear_numpy,
    rasterize_legacy_vertex_heightfield,
    rasterize_surface_height_layers,
    rasterize_top_surface_heightfield,
)


SMPL_BODY_NAMES = (
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
)
SMPL_PARENTS = np.asarray(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12,
     11, 14, 15, 16, 17, 11, 19, 20, 21, 22],
    dtype=np.int64,
)
FOOT_BODY_IDS = np.asarray([3, 4, 7, 8], dtype=np.int64)
BODY_GROUPS = {
    "legs": np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
    "torso": np.asarray([9, 10, 11, 12, 13], dtype=np.int64),
    "arms": np.asarray([14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int64),
    "upper_body": np.asarray(list(range(9, 24)), dtype=np.int64),
}


def _load_motion(path: Path) -> dict[str, Any]:
    motion = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(motion, dict):
        raise TypeError(f"Expected a packaged MotionLib dict: {path}")
    required = {"gts", "grs", "gvs", "gavs", "dps", "dvs", "motion_dt"}
    missing = required.difference(motion)
    if missing:
        raise KeyError(f"{path} is missing MotionLib keys: {sorted(missing)}")
    if int(torch.as_tensor(motion.get("motion_num_frames", [len(motion["gts"])] )).numel()) != 1:
        raise ValueError("This evaluator currently expects a single-motion MotionLib.")
    return motion


def _as_numpy(motion: dict[str, Any], key: str) -> np.ndarray:
    value = motion[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class RasterMeshHeightQuery:
    """Match ProtoMotions MeshTerrain's rasterization and bilinear queries."""

    def __init__(
        self,
        mesh_path: Path,
        horizontal_scale: float = 0.1,
        rasterizer: str = "triangle_surface_layers_v1",
        support_ceiling_tolerance: float = 0.05,
    ):
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                raise ValueError(f"Empty mesh scene: {mesh_path}")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"Expected triangle mesh with faces: {mesh_path}")

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        self.origin_xy = bounds[0, :2]
        self.horizontal_scale = float(horizontal_scale)
        span = bounds[1, :2] - bounds[0, :2]
        rows = max(2, int(np.ceil(span[0] / self.horizontal_scale)) + 1)
        cols = max(2, int(np.ceil(span[1] / self.horizontal_scale)) + 1)
        self.fallback_height = float(bounds[0, 2])
        self.support_ceiling_tolerance = float(support_ceiling_tolerance)
        self.layered = rasterizer == "triangle_surface_layers_v1"
        if self.layered:
            self.height_layers = rasterize_surface_height_layers(
                vertices,
                np.asarray(mesh.faces, dtype=np.int64),
                origin_xy=self.origin_xy,
                shape=(rows, cols),
                horizontal_scale=self.horizontal_scale,
            )
            finite = np.where(np.isfinite(self.height_layers), self.height_layers, -np.inf)
            heights = np.max(finite, axis=2)
            heights[~np.isfinite(heights)] = self.fallback_height
        elif rasterizer == "triangle_surface_v1":
            heights = rasterize_top_surface_heightfield(
                vertices,
                np.asarray(mesh.faces, dtype=np.int64),
                origin_xy=self.origin_xy,
                shape=(rows, cols),
                horizontal_scale=self.horizontal_scale,
                fallback_height=float(bounds[0, 2]),
            )
            self.height_layers = heights[..., None]
        elif rasterizer == "legacy_vertex_griddata":
            heights = rasterize_legacy_vertex_heightfield(
                vertices,
                origin_xy=self.origin_xy,
                shape=(rows, cols),
                horizontal_scale=self.horizontal_scale,
            )
            self.height_layers = heights[..., None]
        else:
            raise ValueError(f"Unsupported terrain rasterizer: {rasterizer!r}")
        self.heights = heights
        self.rasterizer = rasterizer

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        if self.layered:
            return query_height_layers_bilinear_numpy(
                self.height_layers,
                positions,
                origin_xy=self.origin_xy,
                horizontal_scale=self.horizontal_scale,
                support_ceiling_tolerance=self.support_ceiling_tolerance,
                fallback_height=self.fallback_height,
            )
        shape = positions.shape[:-1]
        points = positions[..., :2].reshape(-1, 2)
        grid = (points - self.origin_xy) / self.horizontal_scale
        gx = np.clip(grid[:, 0], 0, self.heights.shape[0] - 1.0001)
        gy = np.clip(grid[:, 1], 0, self.heights.shape[1] - 1.0001)
        x0 = np.minimum(np.floor(gx).astype(np.int64), self.heights.shape[0] - 2)
        y0 = np.minimum(np.floor(gy).astype(np.int64), self.heights.shape[1] - 2)
        fx, fy = gx - x0, gy - y0
        values = (
            self.heights[x0, y0] * (1 - fx) * (1 - fy)
            + self.heights[x0 + 1, y0] * fx * (1 - fy)
            + self.heights[x0, y0 + 1] * (1 - fx) * fy
            + self.heights[x0 + 1, y0 + 1] * fx * fy
        )
        return values.reshape(shape)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    xyz = aw * bv + bw * av + np.cross(av, bv)
    w = aw * bw - np.sum(av * bv, axis=-1, keepdims=True)
    return np.concatenate([xyz, w], axis=-1)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    q = np.where(q[..., 3:4] < 0, -q, q)
    sin_half = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(q[..., 3:4], 1e-12, None))
    axis = q[..., :3] / np.maximum(sin_half, 1e-12)
    return axis * angle


def _quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(_quat_normalize(a) * _quat_normalize(b), axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _global_to_local(global_rot: np.ndarray) -> np.ndarray:
    local = np.empty_like(global_rot)
    local[:, 0] = global_rot[:, 0]
    for body_id in range(1, global_rot.shape[1]):
        parent = SMPL_PARENTS[body_id]
        local[:, body_id] = _quat_mul(
            _quat_conjugate(global_rot[:, parent]), global_rot[:, body_id]
        )
    return _quat_normalize(local)


def _angular_velocity(local_rot: np.ndarray, dt: float) -> np.ndarray:
    delta = _quat_mul(_quat_conjugate(local_rot[:-1]), local_rot[1:])
    return _quat_to_rotvec(delta) / dt


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _norm_stats(values: np.ndarray, prefix: str, scale: float = 1.0) -> dict[str, float]:
    norms = np.linalg.norm(values, axis=-1).reshape(-1) * scale
    if norms.size == 0:
        return {f"{prefix}_rms": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_rms": _rms(norms),
        f"{prefix}_p95": float(np.quantile(norms, 0.95)),
        f"{prefix}_max": float(norms.max()),
    }


def _aligned_slices(pred_pos: np.ndarray, ref_pos: np.ndarray, max_shift: int) -> tuple[slice, slice, int]:
    best: tuple[float, slice, slice, int] | None = None
    for shift in range(-max_shift, max_shift + 1):
        pred_start = max(0, -shift)
        ref_start = max(0, shift)
        count = min(len(pred_pos) - pred_start, len(ref_pos) - ref_start)
        if count < 4:
            continue
        pred = pred_pos[pred_start : pred_start + count]
        ref = ref_pos[ref_start : ref_start + count]
        pred_rel = pred - pred[:, :1]
        ref_rel = ref - ref[:, :1]
        score = np.linalg.norm(pred[:, 0] - ref[:, 0], axis=-1).mean()
        score += np.linalg.norm(pred_rel - ref_rel, axis=-1).mean()
        candidate = (float(score), slice(pred_start, pred_start + count), slice(ref_start, ref_start + count), shift)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Prediction/reference overlap is too short for evaluation.")
    return best[1], best[2], best[3]


def evaluate_prediction(
    prediction: dict[str, Any],
    reference: dict[str, Any],
    terrain: RasterMeshHeightQuery,
    max_shift: int,
    contact_clearance: float,
) -> dict[str, float | int]:
    pred_pos_all = _as_numpy(prediction, "gts").astype(np.float64)
    ref_pos_all = _as_numpy(reference, "gts").astype(np.float64)
    pred_slice, ref_slice, shift = _aligned_slices(pred_pos_all, ref_pos_all, max_shift)

    pred_pos, ref_pos = pred_pos_all[pred_slice], ref_pos_all[ref_slice]
    pred_rot = _as_numpy(prediction, "grs")[pred_slice].astype(np.float64)
    ref_rot = _as_numpy(reference, "grs")[ref_slice].astype(np.float64)
    pred_vel = _as_numpy(prediction, "gvs")[pred_slice].astype(np.float64)
    ref_vel = _as_numpy(reference, "gvs")[ref_slice].astype(np.float64)
    pred_dof_vel = _as_numpy(prediction, "dvs")[pred_slice].astype(np.float64)
    dt = float(np.asarray(_as_numpy(reference, "motion_dt")).reshape(-1)[0])
    n = len(pred_pos)

    metrics: dict[str, float | int] = {
        "reference_shift_frames": int(shift),
        "evaluated_frames": int(n),
        "prediction_frames": int(len(pred_pos_all)),
        "reference_frames": int(len(ref_pos_all)),
        "frame_coverage": float(n / max(len(ref_pos_all), 1)),
        "fps": float(1.0 / dt),
    }

    global_error = np.linalg.norm(pred_pos - ref_pos, axis=-1)
    pred_rel, ref_rel = pred_pos - pred_pos[:, :1], ref_pos - ref_pos[:, :1]
    relative_error = np.linalg.norm(pred_rel - ref_rel, axis=-1)
    root_error = np.linalg.norm(pred_pos[:, 0] - ref_pos[:, 0], axis=-1)
    body_velocity_ratio = float(
        _rms(np.linalg.norm(pred_vel, axis=-1))
        / max(_rms(np.linalg.norm(ref_vel, axis=-1)), 1e-8)
    )
    root_velocity_ratio = float(
        _rms(np.linalg.norm(pred_vel[:, 0], axis=-1))
        / max(_rms(np.linalg.norm(ref_vel[:, 0], axis=-1)), 1e-8)
    )
    metrics.update({
        "global_mpjpe_cm": float(global_error.mean() * 100),
        "global_mpjpe_p95_cm": float(np.quantile(global_error, 0.95) * 100),
        "root_relative_mpjpe_cm": float(relative_error.mean() * 100),
        "root_position_error_cm": float(root_error.mean() * 100),
        "body_velocity_error_m_s": float(np.linalg.norm(pred_vel - ref_vel, axis=-1).mean()),
        "body_velocity_rms_ratio_to_reference": body_velocity_ratio,
        "body_velocity_rms_log_ratio_abs": float(abs(np.log(max(body_velocity_ratio, 1e-8)))),
        "root_velocity_rms_ratio_to_reference": root_velocity_ratio,
        "root_velocity_rms_log_ratio_abs": float(abs(np.log(max(root_velocity_ratio, 1e-8)))),
    })

    pred_local, ref_local = _global_to_local(pred_rot), _global_to_local(ref_rot)
    local_error = _quat_angle(pred_local, ref_local)
    global_rot_error = _quat_angle(pred_rot, ref_rot)
    deg = 180.0 / np.pi
    metrics.update({
        "global_rotation_error_deg": float(global_rot_error.mean() * deg),
        "local_rotation_error_deg": float(local_error[:, 1:].mean() * deg),
        "local_rotation_error_p95_deg": float(np.quantile(local_error[:, 1:], 0.95) * deg),
        "local_rotation_error_gt30_fraction": float((local_error[:, 1:] > np.deg2rad(30)).mean()),
    })
    for group_name, ids in BODY_GROUPS.items():
        metrics[f"{group_name}_local_rotation_error_deg"] = float(local_error[:, ids].mean() * deg)

    pred_angvel, ref_angvel = _angular_velocity(pred_local, dt), _angular_velocity(ref_local, dt)
    pred_angacc = np.diff(pred_angvel, axis=0) / dt
    ref_angacc = np.diff(ref_angvel, axis=0) / dt
    pred_angjerk = np.diff(pred_angacc, axis=0) / dt
    ref_angjerk = np.diff(ref_angacc, axis=0) / dt
    metrics.update(_norm_stats(pred_angvel[:, 1:], "local_angvel_deg_s", deg))
    metrics.update(_norm_stats(pred_angacc[:, 1:], "local_angacc_deg_s2", deg))
    metrics.update(_norm_stats(pred_angjerk[:, 1:], "local_angjerk_deg_s3", deg))
    local_angvel_ratio = float(
        _rms(np.linalg.norm(pred_angvel[:, 1:], axis=-1))
        / max(_rms(np.linalg.norm(ref_angvel[:, 1:], axis=-1)), 1e-8)
    )
    metrics["local_angvel_rms_ratio_to_reference"] = local_angvel_ratio
    metrics["local_angvel_rms_log_ratio_abs"] = float(
        abs(np.log(max(local_angvel_ratio, 1e-8)))
    )
    for group_name, ids in BODY_GROUPS.items():
        group_ids = [index for index in ids if index > 0]
        if group_ids:
            group_ratio = float(
                _rms(np.linalg.norm(pred_angvel[:, group_ids], axis=-1))
                / max(
                    _rms(np.linalg.norm(ref_angvel[:, group_ids], axis=-1)),
                    1e-8,
                )
            )
            metrics[f"{group_name}_angvel_rms_ratio_to_reference"] = group_ratio
            metrics[f"{group_name}_angvel_rms_log_ratio_abs"] = float(
                abs(np.log(max(group_ratio, 1e-8)))
            )
    metrics["local_angacc_rms_ratio_to_reference"] = float(
        _rms(np.linalg.norm(pred_angacc[:, 1:], axis=-1))
        / max(_rms(np.linalg.norm(ref_angacc[:, 1:], axis=-1)), 1e-8)
    )
    metrics["local_angjerk_rms_ratio_to_reference"] = float(
        _rms(np.linalg.norm(pred_angjerk[:, 1:], axis=-1))
        / max(_rms(np.linalg.norm(ref_angjerk[:, 1:], axis=-1)), 1e-8)
    )

    pose_error_q = _quat_mul(_quat_conjugate(ref_local), pred_local)
    pose_error_vec = _quat_to_rotvec(pose_error_q)
    pose_error_rate = np.diff(pose_error_vec, axis=0) / dt
    pose_error_acc = np.diff(pose_error_rate, axis=0) / dt
    metrics.update(_norm_stats(pose_error_rate[:, 1:], "pose_error_rate_deg_s", deg))
    metrics.update(_norm_stats(pose_error_acc[:, 1:], "pose_error_acc_deg_s2", deg))

    body_jerk = np.diff(pred_pos, n=3, axis=0) / (dt ** 3)
    root_jerk = body_jerk[:, 0]
    metrics.update(_norm_stats(body_jerk, "body_jerk_m_s3"))
    metrics.update(_norm_stats(root_jerk, "root_jerk_m_s3"))
    first_count = min(n, max(4, int(round(0.5 / dt))))
    first_jerk = np.diff(pred_pos[:first_count], n=3, axis=0) / (dt ** 3)
    metrics["first_0p5s_body_jerk_rms_m_s3"] = _rms(np.linalg.norm(first_jerk, axis=-1))
    metrics["dof_velocity_rms_rad_s"] = _rms(pred_dof_vel)

    pred_ground = terrain(pred_pos)
    pred_clearance = pred_pos[..., 2] - pred_ground
    penetration = np.maximum(-pred_clearance, 0.0)
    metrics.update({
        "body_origin_penetration_mean_mm": float(penetration.mean() * 1000),
        "body_origin_penetration_max_mm": float(penetration.max() * 1000),
        "penetrating_frame_fraction": float((penetration.max(axis=1) > 0).mean()),
        "foot_origin_penetration_mean_mm": float(penetration[:, FOOT_BODY_IDS].mean() * 1000),
    })

    pred_contacts = prediction.get("contacts")
    if pred_contacts is not None:
        pred_contact = np.asarray(
            pred_contacts.detach().cpu().numpy() if torch.is_tensor(pred_contacts) else pred_contacts,
            dtype=bool,
        )[pred_slice][:, FOOT_BODY_IDS]
    else:
        pred_contact = (
            (pred_clearance[:, FOOT_BODY_IDS] < contact_clearance)
            & (np.abs(pred_vel[:, FOOT_BODY_IDS, 2]) < 0.2)
        )
    foot_speed_xy = np.linalg.norm(pred_vel[:, FOOT_BODY_IDS, :2], axis=-1)
    supported_speed = foot_speed_xy[pred_contact]
    metrics.update({
        "foot_contact_fraction": float(pred_contact.mean()),
        "foot_slide_mean_cm_s": float(supported_speed.mean() * 100) if supported_speed.size else 0.0,
        "foot_slide_p95_cm_s": float(np.quantile(supported_speed, 0.95) * 100) if supported_speed.size else 0.0,
    })

    ref_contacts = reference.get("contacts")
    if ref_contacts is not None:
        ref_contact = np.asarray(
            ref_contacts.detach().cpu().numpy() if torch.is_tensor(ref_contacts) else ref_contacts,
            dtype=bool,
        )[ref_slice][:, FOOT_BODY_IDS]
        metrics["foot_contact_mismatch_fraction"] = float(np.not_equal(pred_contact, ref_contact).mean())
    return metrics


def _parse_prediction(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Prediction must use LABEL=/path/to/predicted.pt")
    label, path = spec.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Prediction label cannot be empty")
    return label, Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=_parse_prediction, required=True)
    parser.add_argument("--terrain-mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--height-grid-scale", type=float, default=0.1)
    parser.add_argument(
        "--terrain-rasterizer",
        choices=(
            "triangle_surface_layers_v1",
            "triangle_surface_v1",
            "legacy_vertex_griddata",
        ),
        default="triangle_surface_layers_v1",
    )
    parser.add_argument("--support-ceiling-tolerance", type=float, default=0.05)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--contact-clearance", type=float, default=0.08)
    args = parser.parse_args()

    reference_path = args.reference.expanduser().resolve()
    terrain_path = args.terrain_mesh.expanduser().resolve()
    reference = _load_motion(reference_path)
    terrain = RasterMeshHeightQuery(
        terrain_path,
        args.height_grid_scale,
        rasterizer=args.terrain_rasterizer,
        support_ceiling_tolerance=args.support_ceiling_tolerance,
    )
    results: dict[str, Any] = {
        "reference": str(reference_path),
        "terrain_mesh": str(terrain_path),
        "terrain_rasterizer": args.terrain_rasterizer,
        "support_ceiling_tolerance": args.support_ceiling_tolerance,
        "body_names": list(SMPL_BODY_NAMES),
        "predictions": {},
    }
    for label, prediction_path in args.prediction:
        metrics = evaluate_prediction(
            _load_motion(prediction_path),
            reference,
            terrain,
            max_shift=args.max_shift,
            contact_clearance=args.contact_clearance,
        )
        results["predictions"][label] = {"path": str(prediction_path), **metrics}

    output = json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"label": label, **metrics} for label, metrics in results["predictions"].items()]
        keys = ["label"] + sorted({key for row in rows for key in row if key not in {"label", "path"}})
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
