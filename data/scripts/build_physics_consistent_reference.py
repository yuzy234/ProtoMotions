#!/usr/bin/env python3
"""Attach source identity metadata to a simulated MotionLib trajectory.

The predicted trajectory supplies every frame-level state. The source library
supplies only per-motion SMPL identity metadata. In particular, source ``lrs``
are never copied because they describe the noisy source pose, not the rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts")
MOTION_METADATA_KEYS = (
    "motion_betas",
    "motion_genders",
    "motion_asset_files",
    "source_global_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _num_motions(data: dict) -> int:
    return int(data["motion_num_frames"].numel())


def build_reference(prediction: dict, source: dict) -> dict:
    num_motions = _num_motions(prediction)
    if num_motions != _num_motions(source):
        raise ValueError(
            "Prediction/source motion count mismatch: "
            f"{num_motions} vs {_num_motions(source)}"
        )
    for key in FRAME_KEYS:
        if key not in prediction:
            raise KeyError(f"Prediction is missing frame field {key!r}")

    out = {}
    for key, value in prediction.items():
        if key == "lrs":
            continue
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        elif isinstance(value, (list, tuple)):
            out[key] = tuple(value)
        else:
            out[key] = value

    for key in MOTION_METADATA_KEYS:
        if key not in source:
            raise KeyError(f"Source is missing required identity field {key!r}")
        value = source[key]
        if torch.is_tensor(value):
            if value.shape[0] != num_motions:
                raise ValueError(f"{key} has incompatible first dimension")
            value = value.detach().cpu().clone()
        elif isinstance(value, (list, tuple)):
            if len(value) != num_motions:
                raise ValueError(f"{key} has incompatible length")
            value = tuple(value)
        out[key] = value

    out["motion_weights"] = torch.ones(num_motions, dtype=torch.float32)
    out["reference_provenance"] = {
        "kind": "physics_consistent_policy_rollout",
        "source_motion_files": tuple(source.get("motion_files", ())),
    }
    return out


def main() -> None:
    args = parse_args()
    prediction = torch.load(args.prediction, map_location="cpu", weights_only=False)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    output = build_reference(prediction, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved {args.output}")
    print(
        f"motions={_num_motions(output)}, "
        f"frames={output['motion_num_frames'].tolist()}, "
        f"source_global_ids={output['source_global_ids'].tolist()}"
    )


if __name__ == "__main__":
    main()
