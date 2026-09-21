#!/usr/bin/env python3
"""Extract one motion from a packaged ProtoMotions MotionLib .pt file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs")
MOTION_KEYS = (
    "motion_lengths",
    "motion_dt",
    "motion_num_frames",
    "motion_weights",
    "motion_betas",
    "motion_genders",
    "source_global_ids",
)
TUPLE_MOTION_KEYS = ("motion_files", "motion_asset_files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--motion-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=False)

    motion_id = args.motion_id
    num_motions = int(data["motion_num_frames"].shape[0])
    if motion_id < 0 or motion_id >= num_motions:
        raise ValueError(f"motion-id {motion_id} out of range [0, {num_motions - 1}]")

    start = int(data["length_starts"][motion_id].item())
    nframes = int(data["motion_num_frames"][motion_id].item())
    stop = start + nframes

    out = {}
    for key, value in data.items():
        if key in FRAME_KEYS:
            out[key] = value[start:stop].clone()
        elif key == "length_starts":
            out[key] = torch.zeros(1, dtype=value.dtype)
        elif key in MOTION_KEYS:
            out[key] = value[motion_id : motion_id + 1].clone()
        elif key in TUPLE_MOTION_KEYS:
            out[key] = (value[motion_id],)
        else:
            out[key] = value

    # A single-motion file should sample only this motion.
    out["motion_weights"] = torch.ones_like(out["motion_weights"], dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    source_gid = out.get("source_global_ids")
    source_gid_str = (
        str(int(source_gid[0].item())) if torch.is_tensor(source_gid) else "unknown"
    )
    print(f"Saved {args.output}")
    print(f"motion_id={motion_id}, source_global_id={source_gid_str}, frames={nframes}")


if __name__ == "__main__":
    main()
