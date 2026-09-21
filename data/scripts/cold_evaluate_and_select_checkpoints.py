# SPDX-License-Identifier: Apache-2.0
"""Cold-evaluate saved tracker checkpoints and select a deployment model."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"epoch_(\d+)\.ckpt", path.name)
    if match is None:
        raise ValueError(f"Unsupported checkpoint name: {path.name}")
    return int(match.group(1))


def _run(command: list[str], cwd: Path, log_path: Path | None = None) -> None:
    env = os.environ.copy()
    current_python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(cwd), current_python_path) if entry
    )
    # Isaac Gym loads libpython dynamically.  Make the current conda
    # environment self-contained even when this script is launched without a
    # preceding ``conda activate``.
    python_lib = Path(sys.executable).resolve().parent.parent / "lib"
    current_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        entry for entry in (str(python_lib), current_library_path) if entry
    )
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--terrain-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--selected-checkpoint", type=Path, default=None)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--simulator", default="isaacgym")
    parser.add_argument("--epoch-start", type=int, default=None)
    parser.add_argument("--epoch-stop", type=int, default=None)
    parser.add_argument("--epoch-stride", type=int, default=1)
    parser.add_argument("--tracking-relative-tolerance", type=float, default=0.15)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--shape-asset-dir", type=Path, default=None)
    parser.add_argument("--smpl-data-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    motion_file = args.motion_file.expanduser().resolve()
    terrain_mesh = args.terrain_mesh.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_dir / "cold_eval"
    )
    selected_checkpoint = (
        args.selected_checkpoint.expanduser().resolve()
        if args.selected_checkpoint is not None
        else checkpoint_dir / "cold_selected.ckpt"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.ckpt"), key=_checkpoint_epoch
    )
    checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if (args.epoch_start is None or _checkpoint_epoch(checkpoint) >= args.epoch_start)
        and (args.epoch_stop is None or _checkpoint_epoch(checkpoint) <= args.epoch_stop)
        and (
            _checkpoint_epoch(checkpoint) == args.epoch_start
            or _checkpoint_epoch(checkpoint) % args.epoch_stride == 0
        )
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")

    predictions: list[tuple[str, Path]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        epoch = _checkpoint_epoch(checkpoint)
        label = f"epoch_{epoch}"
        prediction = checkpoint_dir / "results" / f"predicted_motion_lib_epoch_{epoch}.pt"
        log_path = output_dir / f"epoch_{epoch}.log"
        predictions.append((label, prediction))
        if not args.force and prediction.exists() and log_path.exists():
            print(f"[{index}/{len(checkpoints)}] reuse cold rollout {label}", flush=True)
            continue
        print(f"[{index}/{len(checkpoints)}] cold rollout {label}", flush=True)
        command = [
            sys.executable,
            "protomotions/inference_agent.py",
            "--checkpoint", str(checkpoint),
            "--motion-file", str(motion_file),
            "--terrain-mesh", str(terrain_mesh),
            "--simulator", args.simulator,
            "--num-envs", "1",
            "--headless",
            "--full-eval",
            "--motion-id", str(args.motion_id),
            "--preserve-reference-world-position",
            "--smpl-shape-from-motion",
            "--save-predicted-motion-lib",
        ]
        if args.shape_asset_dir is not None:
            command.extend(["--shape-asset-dir", str(args.shape_asset_dir)])
        if args.smpl_data_dir is not None:
            command.extend(["--smpl-data-dir", str(args.smpl_data_dir)])
        _run(command, repo, log_path)
        if not prediction.exists():
            raise FileNotFoundError(
                f"Inference completed but did not create prediction: {prediction}"
            )

    quality_json = output_dir / "all_epochs_quality.json"
    quality_csv = output_dir / "all_epochs_quality.csv"
    quality_command = [
        sys.executable,
        "data/scripts/evaluate_physical_motion_quality.py",
        "--reference", str(motion_file),
        "--terrain-mesh", str(terrain_mesh),
    ]
    for label, prediction in predictions:
        quality_command.extend(["--prediction", f"{label}={prediction}"])
    quality_command.extend(
        ["--output-json", str(quality_json), "--output-csv", str(quality_csv)]
    )
    _run(quality_command, repo, output_dir / "physical_quality.log")

    selection_command = [
        sys.executable,
        "data/scripts/select_physical_motion_checkpoint.py",
        "--quality-json", str(quality_json),
        "--checkpoint-dir", str(checkpoint_dir),
        "--cold-log-dir", str(output_dir),
        "--output-json", str(output_dir / "checkpoint_selection.json"),
        "--output-csv", str(output_dir / "checkpoint_selection.csv"),
        "--deploy-checkpoint", str(selected_checkpoint),
        "--tracking-relative-tolerance", str(args.tracking_relative_tolerance),
        "--min-success-rate", str(args.min_success_rate),
    ]
    _run(selection_command, repo)
    print(f"Cold-selected checkpoint: {selected_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
