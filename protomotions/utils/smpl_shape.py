"""Utilities for generating SMPL MJCF assets from per-motion shape betas.

This follows the same SMPLSim/CRISP path used by CRISP MotionTracking:
``SMPL_Robot.load_from_skeleton(betas=..., gender=...)`` followed by
``write_xml``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import torch


DEFAULT_SMPL_DATA_DIR_CANDIDATES = (
    "data/smpl",
    "/home/yzy/part1/workspace/test/baseline/CRISP-Real2Sim/prep/Contact-Predictor/data/body_models/smpl",
    "/home/yzy/part1/workspace/test/baseline/CRISP-Real2Sim/prep/HMR/inputs/checkpoints/body_models/smpl",
    "/home/yzy/part1/workspace/test/baseline/CRISP-Real2Sim/prep/data/smpl",
)


CRISP_SMPL_ROBOT_CFG = {
    "mesh": False,
    "replace_feet": True,
    "rel_joint_lm": False,
    "remove_toe": False,
    "freeze_hand": False,
    "real_weight_porpotion_capsules": True,
    "real_weight_porpotion_boxes": True,
    "real_weight": True,
    "master_range": 30,
    "big_ankle": True,
    "box_body": True,
    "masterfoot": False,
    "upright_start": True,
    "model": "smpl",
    "create_vel_sensors": False,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
    "gender": "neutral",
}


def resolve_smpl_data_dir(smpl_data_dir: Optional[str] = None) -> str:
    candidates = []
    if smpl_data_dir:
        candidates.append(smpl_data_dir)
    candidates.extend(DEFAULT_SMPL_DATA_DIR_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists() and (path / "SMPL_NEUTRAL.pkl").exists():
            return str(path.resolve())

    raise FileNotFoundError(
        "Could not find SMPL model data directory. Pass --smpl-data-dir pointing "
        "to a directory containing SMPL_NEUTRAL.pkl / SMPL_MALE.pkl / SMPL_FEMALE.pkl."
    )


def normalize_betas(betas) -> torch.Tensor:
    betas_t = torch.as_tensor(betas, dtype=torch.float32).flatten()
    if betas_t.numel() < 10:
        betas_t = torch.cat([betas_t, torch.zeros(10 - betas_t.numel())])
    elif betas_t.numel() > 10:
        betas_t = betas_t[:10]
    return betas_t


def shape_asset_name(betas, gender: int = 0, prefix: str = "smpl_humanoid_shape") -> str:
    betas_t = normalize_betas(betas)
    payload = betas_t.numpy().astype("float32").tobytes() + int(gender).to_bytes(
        1, "little", signed=False
    )
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"{prefix}_g{int(gender)}_{digest}.xml"


def generate_smpl_humanoid_xml(
    betas,
    output_path: str | os.PathLike,
    gender: int = 0,
    smpl_data_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return str(output_path.resolve())

    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

    data_dir = resolve_smpl_data_dir(smpl_data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    smpl_robot = SMPL_Robot(CRISP_SMPL_ROBOT_CFG, data_dir=data_dir)
    smpl_robot.load_from_skeleton(
        betas=normalize_betas(betas).view(1, -1),
        gender=torch.tensor([int(gender)], dtype=torch.long),
        objs_info=None,
    )
    smpl_robot.write_xml(str(output_path))
    return str(output_path.resolve())


def generate_smpl_humanoid_xml_for_shape(
    betas,
    output_dir: str | os.PathLike,
    gender: int = 0,
    smpl_data_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    output_dir = Path(output_dir)
    asset_name = shape_asset_name(betas, gender=gender)
    return generate_smpl_humanoid_xml(
        betas=betas,
        output_path=output_dir / asset_name,
        gender=gender,
        smpl_data_dir=smpl_data_dir,
        overwrite=overwrite,
    )


def configure_robot_from_motion_shape(
    robot_config,
    motion_file: str | os.PathLike,
    motion_id: int,
    shape_asset_dir: str | os.PathLike,
    smpl_data_dir: Optional[str] = None,
    preserve_control_info: bool = False,
) -> str:
    """Configure one SMPL simulator robot from a packaged motion's betas.

    A simulator instance can contain one humanoid asset, so this helper is
    intentionally for a fixed-motion run.  It is shared by training and
    inference to guarantee both use the same collision geometry, masses,
    kinematics, and PD metadata.
    """
    from protomotions.components.pose_lib import extract_kinematic_info

    motion_path = Path(motion_file).expanduser()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file does not exist: {motion_path}")

    motion_data = torch.load(motion_path, map_location="cpu", weights_only=False)
    if "motion_betas" not in motion_data:
        raise KeyError(
            f"{motion_path} does not contain 'motion_betas'. Re-run motion conversion "
            "with --per-motion-smpl-shape."
        )
    num_motions = len(motion_data["motion_betas"])
    if motion_id < 0 or motion_id >= num_motions:
        raise IndexError(
            f"motion_id={motion_id} is outside [0, {num_motions - 1}] for {motion_path}"
        )

    betas = motion_data["motion_betas"][motion_id]
    genders = motion_data.get(
        "motion_genders", torch.zeros(num_motions, dtype=torch.long)
    )
    gender = int(genders[motion_id].item())

    asset_file = None
    motion_asset_files = motion_data.get("motion_asset_files")
    if motion_asset_files is not None and motion_id < len(motion_asset_files):
        candidate = Path(str(motion_asset_files[motion_id])).expanduser()
        if candidate.exists():
            asset_file = str(candidate.resolve())

    if asset_file is None:
        asset_file = generate_smpl_humanoid_xml_for_shape(
            betas=betas,
            output_dir=shape_asset_dir,
            gender=gender,
            smpl_data_dir=smpl_data_dir,
        )

    # A checkpoint's resolved control parameters are part of the learned
    # dynamics. Replacing only the body shape during inference must not
    # silently replace those parameters with MJCF defaults.
    previous_control_info = None
    if preserve_control_info and hasattr(robot_config.control, "control_info"):
        previous_control_info = dict(robot_config.control.control_info)

    asset_path = Path(asset_file).resolve()
    robot_config.asset.asset_root = str(asset_path.parent)
    robot_config.asset.asset_file_name = asset_path.name
    robot_config.kinematic_info = extract_kinematic_info(str(asset_path))
    if hasattr(robot_config.control, "control_info"):
        delattr(robot_config.control, "control_info")
    robot_config.control.initialize_control_info(robot_config.asset)
    if previous_control_info is not None:
        for dof_name, control_info in previous_control_info.items():
            if dof_name in robot_config.control.control_info:
                robot_config.control.control_info[dof_name] = control_info
    robot_config.number_of_actions = robot_config.kinematic_info.num_dofs
    if robot_config.anchor_body_name is None:
        robot_config.anchor_body_index = 0
    else:
        robot_config.anchor_body_index = robot_config.kinematic_info.body_names.index(
            robot_config.anchor_body_name
        )
    if (
        robot_config.default_dof_pos is None
        or robot_config.default_dof_pos.numel() != robot_config.number_of_actions
    ):
        robot_config.default_dof_pos = torch.zeros(
            robot_config.number_of_actions, dtype=torch.float32
        )
    return str(asset_path)
