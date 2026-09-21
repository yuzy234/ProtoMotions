#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert EasyMimic Isaac-Z-up SMPL tracks to a packaged ProtoMotions MotionLib."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciPyRotation

from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from protomotions.components.pose_lib import (
    compute_angular_velocity,
    compute_forward_kinematics_from_transforms,
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.terrains.mesh_terrain import MeshTerrain
from protomotions.utils.smpl_shape import (
    generate_smpl_humanoid_xml_for_shape,
    resolve_smpl_data_dir,
)
from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_mul,
    quaternion_to_matrix,
)
from data.scripts.robustify_motion_velocities import robustify_motion_velocities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pt", type=Path)
    parser.add_argument("output_pt", type=Path)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--global-id",
        type=int,
        action="append",
        help="Global ID to include. Repeat for multiple IDs; default includes all.",
    )
    parser.add_argument(
        "--height-query-resolution",
        type=float,
        default=0.05,
        help="Raster resolution used only for contact labels.",
    )
    parser.add_argument(
        "--per-motion-smpl-shape",
        action="store_true",
        help="Generate a CRISP/SMPLSim MJCF per motion from that motion's betas and use it for FK.",
    )
    parser.add_argument(
        "--shape-asset-dir",
        type=Path,
        default=Path("data/easymimic/assets/mjcf"),
        help="Directory for generated per-shape SMPL MJCF files.",
    )
    parser.add_argument(
        "--smpl-data-dir",
        type=Path,
        default=None,
        help="Directory containing SMPL model PKLs. Defaults to known local CRISP paths.",
    )
    parser.add_argument(
        "--velocity-filter-window",
        type=int,
        default=0,
        help=(
            "Optional odd Savitzky-Golay window for gvs/gavs/dvs. "
            "0 disables filtering; poses, world path and contacts are unchanged."
        ),
    )
    parser.add_argument(
        "--velocity-filter-polyorder",
        type=int,
        default=2,
        help="Polynomial order used by --velocity-filter-window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally keep only the first N contiguous frames of each track.",
    )
    parser.add_argument(
        "--surface-contact-labels",
        action="store_true",
        help=(
            "Infer contact from cached 6890-vertex human surfaces instead of only "
            "rigid-body origins. Requires body_vertices_local in the input."
        ),
    )
    parser.add_argument("--surface-contact-distance", type=float, default=0.08)
    parser.add_argument("--surface-contact-penetration", type=float, default=0.08)
    parser.add_argument("--surface-contact-speed", type=float, default=1.0)
    parser.add_argument("--surface-impact-distance", type=float, default=0.02)
    parser.add_argument("--surface-contact-vertices", type=int, default=10)
    parser.add_argument(
        "--copy-confidence-from",
        type=Path,
        default=None,
        help=(
            "Copy root reliability metadata from a geometrically identical "
            "MotionLib while regenerating contact labels."
        ),
    )
    return parser.parse_args()


def load_smpl_vertex_body_ids(smpl_data_dir: Path | None) -> torch.Tensor:
    """Map each native-SMPL vertex to ProtoMotions' MuJoCo body order."""
    model_dir = Path(resolve_smpl_data_dir(
        None if smpl_data_dir is None else str(smpl_data_dir)
    ))
    with (model_dir / "SMPL_NEUTRAL.pkl").open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    weights = np.asarray(model["weights"])
    if weights.shape != (6890, 24):
        raise ValueError(f"Unexpected SMPL skinning-weight shape: {weights.shape}")
    dominant_smpl_joint = weights.argmax(axis=1)
    mujoco_index = {name: index for index, name in enumerate(SMPL_MUJOCO_NAMES)}
    smpl_to_mujoco = np.asarray(
        [mujoco_index[name] for name in SMPL_BONE_ORDER_NAMES], dtype=np.int64
    )
    return torch.from_numpy(smpl_to_mujoco[dominant_smpl_joint]).long()


def cached_vertices_to_world(
    payload: dict,
    mask: torch.Tensor,
    order: torch.Tensor,
    pose_world: torch.Tensor,
    trans_world: torch.Tensor,
) -> torch.Tensor:
    """Transform the cached posed surface with the same root convention as joints."""
    cached = payload.get("body_vertices_local")
    if not isinstance(cached, torch.Tensor):
        raise KeyError(
            "--surface-contact-labels requires body_vertices_local in the input"
        )
    local = cached[mask][order].detach().cpu().numpy()
    if local.ndim != 3 or local.shape[1:] != (6890, 3):
        raise ValueError(f"Expected cached SMPL vertices [T,6890,3], got {local.shape}")
    root_rotation = SciPyRotation.from_rotvec(
        pose_world[:, :3].detach().cpu().numpy()
    ).as_matrix()
    world = np.einsum("tij,tvj->tvi", root_rotation, local)
    world += trans_world.detach().cpu().numpy()[:, None, :]
    return torch.from_numpy(world).to(torch.float32)


def infer_surface_contacts(
    world_vertices: torch.Tensor,
    vertex_body_ids: torch.Tensor,
    terrain: MeshTerrain,
    *,
    fps: float,
    num_bodies: int,
    contact_distance: float,
    penetration_tolerance: float,
    speed_threshold: float,
    impact_distance: float,
    robust_vertex_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer body contact from robust closest human-surface vertices.

    A low-speed near-surface set represents support.  A very close high-speed
    set is retained as an impact event, which is useful for landing and vault
    transitions but does not label a merely nearby swinging limb.
    """
    if fps <= 0 or robust_vertex_count <= 0:
        raise ValueError("fps and robust_vertex_count must be positive")
    if world_vertices.shape[1] != vertex_body_ids.numel():
        raise ValueError("cached vertices and SMPL vertex/body map differ")
    ground = terrain.get_ground_heights(world_vertices)
    clearance = world_vertices[..., 2] - ground
    velocity = torch.zeros_like(world_vertices)
    dt = 1.0 / float(fps)
    velocity[1:-1] = (world_vertices[2:] - world_vertices[:-2]) / (2.0 * dt)
    velocity[0] = (world_vertices[1] - world_vertices[0]) / dt
    velocity[-1] = (world_vertices[-1] - world_vertices[-2]) / dt
    speed = torch.linalg.vector_norm(velocity, dim=-1)

    frame_count = world_vertices.shape[0]
    contacts = torch.zeros(frame_count, num_bodies, dtype=torch.bool)
    robust_clearance = torch.full((frame_count, num_bodies), torch.inf)
    robust_speed = torch.full((frame_count, num_bodies), torch.inf)
    for body_id in range(num_bodies):
        vertex_ids = torch.nonzero(vertex_body_ids == body_id).flatten()
        if vertex_ids.numel() == 0:
            continue
        count = min(int(robust_vertex_count), int(vertex_ids.numel()))
        body_clearance = clearance[:, vertex_ids]
        closest, closest_indices = torch.topk(
            body_clearance, count, dim=1, largest=False
        )
        body_speed = speed[:, vertex_ids].gather(1, closest_indices)
        body_clearance_robust = closest.median(dim=1).values
        body_speed_robust = body_speed.median(dim=1).values
        robust_clearance[:, body_id] = body_clearance_robust
        robust_speed[:, body_id] = body_speed_robust
        near_surface = (
            (body_clearance_robust <= float(contact_distance))
            & (body_clearance_robust >= -float(penetration_tolerance))
        )
        supported = near_surface & (body_speed_robust <= float(speed_threshold))
        impact = (
            body_clearance_robust.abs() <= float(impact_distance)
        )
        contacts[:, body_id] = supported | impact
    return contacts, robust_clearance, robust_speed


def convert_track(
    pose_aa: torch.Tensor,
    root_pos: torch.Tensor,
    fps: float,
    kinematic_info,
    terrain: MeshTerrain,
):
    """Match the checkpoint's SMPL/MJCF convention at an explicit world-space root."""
    pose_np = pose_aa.detach().cpu().numpy()
    num_frames = pose_np.shape[0]
    smpl_to_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(name)
        for name in SMPL_MUJOCO_NAMES
        if name in SMPL_BONE_ORDER_NAMES
    ]

    # ProtoMotions' SMPL asset freezes the final two hand bodies.
    pose_np = np.concatenate(
        [pose_np[:, :66], np.zeros((num_frames, 6), dtype=pose_np.dtype)], axis=1
    )
    pose_np = pose_np.reshape(num_frames, 24, 3)[:, smpl_to_mujoco]
    local_quat = (
        SciPyRotation.from_rotvec(pose_np.reshape(-1, 3))
        .as_quat()
        .reshape(num_frames, 24, 4)
    )

    dtype = torch.float32
    local_quat = torch.from_numpy(local_quat).to(dtype=dtype)
    root_pos = root_pos.to(dtype=dtype, device="cpu")
    local_rot_mats = quaternion_to_matrix(local_quat, w_last=True)

    # This is the SMPL-body-frame correction used by ProtoMotions' AMASS converter.
    _, world_rot_mats = compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos, local_rot_mats
    )
    frame_correction = SciPyRotation.from_euler(
        "xyz", np.array([-np.pi / 2, -np.pi / 2, 0]), degrees=False
    )
    correction_quat = (
        torch.from_numpy(frame_correction.as_quat())
        .to(dtype=dtype)
        .expand(num_frames, -1)
    )
    global_quat = matrix_to_quaternion(world_rot_mats, w_last=True)
    for body_id in range(24):
        global_quat[:, body_id] = quat_mul(
            global_quat[:, body_id], correction_quat, w_last=True
        )
    corrected_local_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=corrected_local_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    motion.local_rigid_body_rot = matrix_to_quaternion(
        corrected_local_rot_mats, w_last=True
    )

    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=corrected_local_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]
    motion.dof_vel = compute_angular_velocity(
        batched_robot_rot_mats=corrected_local_rot_mats[:, 1:],
        fps=fps,
    ).reshape(num_frames, -1)

    ground = terrain.get_ground_heights(motion.rigid_body_pos)
    speed = torch.linalg.norm(motion.rigid_body_vel, dim=-1)
    clearance = motion.rigid_body_pos[..., 2] - ground
    motion.rigid_body_contacts = (speed < 0.15) & (clearance < 0.1)
    return motion


def resolve_world_pelvis(
    payload: dict,
    mask: torch.Tensor,
    order: torch.Tensor,
    pose_aa: torch.Tensor,
    trans_world: torch.Tensor,
    kinematic_info,
) -> tuple[torch.Tensor, str]:
    """Resolve the physics-root world position without mixing body conventions.

    Current GVHMR outputs cache SMPL-X geometry root-local.  Their
    ``trans_world`` is the placement anchor for that cached geometry, not the
    native-SMPL model translation consumed by the legacy converter.  In that
    case the intended root is the cached pelvis transformed into world space.
    Legacy native-SMPL payloads do not contain cached joints and retain the old
    SMPL-model-translation convention.
    """
    cached_joints = payload.get("body_joints_smpl_local")
    if isinstance(cached_joints, torch.Tensor):
        pelvis_local = cached_joints[mask][order, 0].detach().cpu().numpy()
        root_rotation = SciPyRotation.from_rotvec(
            pose_aa[:, :3].detach().cpu().numpy()
        )
        pelvis_world = root_rotation.apply(pelvis_local) + trans_world.detach().cpu().numpy()
        return torch.from_numpy(pelvis_world).to(torch.float32), "cached_smplx_pelvis"

    root_pos = trans_world.to(dtype=torch.float32, device="cpu")
    root_pos = root_pos + kinematic_info.local_pos[0].to(
        dtype=torch.float32, device="cpu"
    )
    return root_pos, "legacy_native_smpl_translation"


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    required = {"global_id", "frame_idx", "pose_world", "trans_world"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"EasyMimic file is missing fields: {sorted(missing)}")
    if payload["pose_world"].shape[-1] != 72:
        raise ValueError(
            f"Expected SMPL axis-angle [T,72], got {tuple(payload['pose_world'].shape)}"
        )

    all_ids = torch.unique(payload["global_id"]).tolist()
    selected_ids = args.global_id if args.global_id else all_ids
    unknown = sorted(set(selected_ids).difference(all_ids))
    if unknown:
        raise ValueError(f"Requested global IDs not found: {unknown}; available={all_ids}")

    terrain_cfg = TerrainConfig(
        mesh_path=str(args.mesh_path),
        horizontal_scale=args.height_query_resolution,
    )
    terrain = MeshTerrain(terrain_cfg, num_envs=1, device=torch.device("cpu"))
    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    ).to(torch.device("cpu"), torch.float32)

    motions = []
    surface_contact_clearances = []
    surface_contact_speeds = []
    motion_names = []
    motion_betas = []
    motion_genders = []
    motion_asset_files = []
    source_global_ids = []
    root_translation_sources = []
    vertex_body_ids = (
        load_smpl_vertex_body_ids(args.smpl_data_dir)
        if args.surface_contact_labels
        else None
    )
    for global_id in selected_ids:
        mask = payload["global_id"] == global_id
        frames = payload["frame_idx"][mask]
        order = torch.argsort(frames)
        if args.max_frames is not None:
            if args.max_frames < 2:
                raise ValueError("--max-frames must be at least two")
            order = order[: args.max_frames]
        frames = frames[order]
        if len(frames) < 2:
            raise ValueError(f"global_id={global_id} has fewer than two frames")
        if not torch.all(frames[1:] - frames[:-1] == 1):
            raise ValueError(f"global_id={global_id} is not a contiguous clip")

        if "betas" in payload:
            betas_seq = payload["betas"][mask][order].to(torch.float32)
            betas = betas_seq.median(dim=0).values
            beta_drift = (betas_seq - betas).abs().max().item()
        else:
            betas = torch.zeros(10, dtype=torch.float32)
            beta_drift = 0.0

        gender = 0  # EasyMimic export has no gender field; use neutral SMPL.
        track_kinematic_info = kinematic_info
        asset_file = "protomotions/data/assets/mjcf/smpl_humanoid.xml"
        if args.per_motion_smpl_shape:
            asset_file = generate_smpl_humanoid_xml_for_shape(
                betas=betas,
                output_dir=args.shape_asset_dir,
                gender=gender,
                smpl_data_dir=str(args.smpl_data_dir) if args.smpl_data_dir else None,
            )
            track_kinematic_info = extract_kinematic_info(asset_file).to(
                torch.device("cpu"), torch.float32
            )

        pose_track = payload["pose_world"][mask][order]
        trans_track = payload["trans_world"][mask][order]
        root_pos, root_translation_source = resolve_world_pelvis(
            payload,
            mask,
            order,
            pose_track,
            trans_track,
            track_kinematic_info,
        )
        motion = convert_track(
            pose_track,
            root_pos,
            args.fps,
            track_kinematic_info,
            terrain,
        )
        if args.surface_contact_labels:
            world_vertices = cached_vertices_to_world(
                payload,
                mask,
                order,
                pose_track,
                trans_track,
            )
            surface_contacts, surface_clearance, surface_speed = (
                infer_surface_contacts(
                    world_vertices,
                    vertex_body_ids,
                    terrain,
                    fps=args.fps,
                    num_bodies=motion.rigid_body_pos.shape[1],
                    contact_distance=args.surface_contact_distance,
                    penetration_tolerance=args.surface_contact_penetration,
                    speed_threshold=args.surface_contact_speed,
                    impact_distance=args.surface_impact_distance,
                    robust_vertex_count=args.surface_contact_vertices,
                )
            )
            motion.rigid_body_contacts = surface_contacts
            surface_contact_clearances.append(surface_clearance)
            surface_contact_speeds.append(surface_speed)
        motions.append(motion)
        motion_names.append(f"global_id_{global_id}")
        motion_betas.append(betas)
        motion_genders.append(gender)
        motion_asset_files.append(asset_file)
        source_global_ids.append(int(global_id))
        root_translation_sources.append(root_translation_source)
        print(
            f"Converted global_id={global_id}: frames={len(frames)}, "
            f"source_frames={int(frames[0])}-{int(frames[-1])}, "
            f"duration={(len(frames) - 1) / args.fps:.3f}s, "
            f"shape={'per-motion' if args.per_motion_smpl_shape else 'neutral'}, "
            f"root={root_translation_source}, "
            f"contact={'surface' if args.surface_contact_labels else 'body-origin'}, "
            f"beta_drift_max={beta_drift:.6g}"
        )

    frame_counts = torch.tensor(
        [motion.motion_num_frames for motion in motions], dtype=torch.long
    )
    length_starts = frame_counts.roll(1)
    length_starts[0] = 0
    length_starts = length_starts.cumsum(0)
    output = {
        "gts": torch.cat([m.rigid_body_pos for m in motions]).float(),
        "grs": torch.cat([m.rigid_body_rot for m in motions]).float(),
        "gvs": torch.cat([m.rigid_body_vel for m in motions]).float(),
        "gavs": torch.cat([m.rigid_body_ang_vel for m in motions]).float(),
        "dvs": torch.cat([m.dof_vel for m in motions]).float(),
        "dps": torch.cat([m.dof_pos for m in motions]).float(),
        "contacts": torch.cat([m.rigid_body_contacts for m in motions]).bool(),
        "lrs": torch.cat([m.local_rigid_body_rot for m in motions]).float(),
        "length_starts": length_starts,
        "motion_lengths": torch.tensor(
            [(m.motion_num_frames - 1) / args.fps for m in motions],
            dtype=torch.float32,
        ),
        "motion_dt": torch.full(
            (len(motions),), 1.0 / args.fps, dtype=torch.float32
        ),
        "motion_num_frames": frame_counts,
        "motion_weights": torch.ones(len(motions), dtype=torch.float32),
        "motion_files": tuple(motion_names),
        "motion_betas": torch.stack(motion_betas).float(),
        "motion_genders": torch.tensor(motion_genders, dtype=torch.long),
        "motion_asset_files": tuple(motion_asset_files),
        "source_global_ids": torch.tensor(source_global_ids, dtype=torch.long),
        "root_translation_sources": tuple(root_translation_sources),
        "contact_label_method": (
            "cached_smpl_surface_v1"
            if args.surface_contact_labels
            else "rigid_body_origin_v1"
        ),
    }
    if surface_contact_clearances:
        output["surface_contact_clearance"] = torch.cat(
            surface_contact_clearances
        ).float()
        output["surface_contact_speed"] = torch.cat(
            surface_contact_speeds
        ).float()
        output["surface_contact_config"] = {
            "distance_m": float(args.surface_contact_distance),
            "penetration_tolerance_m": float(args.surface_contact_penetration),
            "speed_m_s": float(args.surface_contact_speed),
            "impact_distance_m": float(args.surface_impact_distance),
            "robust_vertex_count": int(args.surface_contact_vertices),
        }
    if args.copy_confidence_from is not None:
        confidence_motion = torch.load(
            args.copy_confidence_from, map_location="cpu", weights_only=False
        )
        for geometry_key in ("gts", "grs"):
            candidate = torch.as_tensor(confidence_motion[geometry_key])
            if candidate.shape != output[geometry_key].shape or not torch.allclose(
                candidate, output[geometry_key], atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    "--copy-confidence-from is not geometrically identical: "
                    f"{geometry_key} differs"
                )
        for key in (
            "root_reference_reliability",
            "root_observation_reliability",
            "root_physical_reliability",
            "root_translation_correction",
            "root_projection_metadata",
        ):
            if key in confidence_motion:
                output[key] = confidence_motion[key]
        output["contact_confidence_source_motion"] = str(
            args.copy_confidence_from.expanduser().resolve()
        )
    if args.velocity_filter_window:
        output, velocity_report = robustify_motion_velocities(
            output,
            window_length=args.velocity_filter_window,
            polyorder=args.velocity_filter_polyorder,
        )
        for key, metrics in velocity_report.items():
            print(
                f"Velocity filter {key}: frame_delta_rms "
                f"{metrics['frame_delta_rms_before']:.6f} -> "
                f"{metrics['frame_delta_rms_after']:.6f}"
            )
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_pt)
    print(f"Saved packaged MotionLib: {args.output_pt}")
    print(f"motions={len(motions)}, fps={args.fps}, frames={frame_counts.tolist()}")


if __name__ == "__main__":
    main()
