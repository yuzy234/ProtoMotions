# SPDX-License-Identifier: Apache-2.0
"""Select a motion-tracking checkpoint from cold-rollout quality metrics.

The selector is deliberately scale-free: it first rejects checkpoints that
trade away too much tracking accuracy, then averages percentile ranks over
tracking and physical-quality metrics.  Consequently, a new motion does not
need motion-specific metric normalization or learning-rate tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


TRACKING_METRICS = (
    "global_mpjpe_cm",
    "local_rotation_error_deg",
)
PHYSICAL_METRICS = (
    "body_jerk_m_s3_rms",
    "first_0p5s_body_jerk_rms_m_s3",
    "local_angjerk_deg_s3_rms",
    "foot_slide_mean_cm_s",
    "root_jerk_m_s3_rms",
)
DYNAMIC_FIDELITY_METRICS = (
    "body_velocity_rms_log_ratio_abs",
    "local_angvel_rms_log_ratio_abs",
    "upper_body_angvel_rms_log_ratio_abs",
)
ALL_METRICS = TRACKING_METRICS + PHYSICAL_METRICS


def _epoch_from_label(label: str) -> int:
    match = re.search(r"(?:^|_)epoch[_-]?(\d+)$", label)
    if match is None:
        match = re.search(r"(\d+)$", label)
    if match is None:
        raise ValueError(f"Cannot infer epoch from prediction label: {label!r}")
    return int(match.group(1))


def _read_cold_log(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float] = {}
    patterns = {
        "success_rate": r"eval/success_rate:\s*([-+0-9.eE]+)",
        "evaluator_quality_score": r"Overall Score:\s*([-+0-9.eE]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is not None:
            result[key] = float(match.group(1))
    return result


def _percentile_ranks(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Return average-tie ranks normalized to [0, 1], where zero is best."""
    ordered = sorted(rows, key=lambda row: row[metric])
    denominator = max(len(ordered) - 1, 1)
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        value = ordered[index][metric]
        while end < len(ordered) and math.isclose(
            ordered[end][metric], value, rel_tol=1e-12, abs_tol=1e-12
        ):
            end += 1
        average_rank = 0.5 * (index + end - 1) / denominator
        for tied_index in range(index, end):
            ranks[ordered[tied_index]["label"]] = average_rank
        index = end
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cold-log-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--deploy-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--tracking-relative-tolerance",
        type=float,
        default=0.15,
        help="Keep checkpoints within this fraction of the best cold tracking metrics.",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.99,
        help="Reject checkpoints below this cold-rollout success rate when logs are supplied.",
    )
    args = parser.parse_args()

    quality_path = args.quality_json.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    payload = json.loads(quality_path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", {})
    if not predictions:
        raise ValueError(f"No predictions found in {quality_path}")

    rows: list[dict[str, Any]] = []
    use_dynamic_fidelity = all(
        all(metric in metrics for metric in DYNAMIC_FIDELITY_METRICS)
        for metrics in predictions.values()
    )
    ranking_metrics = ALL_METRICS + (
        DYNAMIC_FIDELITY_METRICS if use_dynamic_fidelity else ()
    )
    for label, metrics in predictions.items():
        epoch = _epoch_from_label(label)
        row: dict[str, Any] = {"label": label, "epoch": epoch}
        for metric in ranking_metrics:
            if metric not in metrics:
                raise KeyError(f"Prediction {label!r} is missing metric {metric!r}")
            value = float(metrics[metric])
            if not math.isfinite(value):
                raise ValueError(f"Prediction {label!r} has non-finite {metric}: {value}")
            row[metric] = value
        checkpoint = checkpoint_dir / f"epoch_{epoch}.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {checkpoint}")
        row["checkpoint"] = str(checkpoint)
        if args.cold_log_dir is not None:
            row.update(
                _read_cold_log(
                    args.cold_log_dir.expanduser().resolve() / f"epoch_{epoch}.log"
                )
            )
        rows.append(row)

    if any("success_rate" in row for row in rows):
        successful = [
            row
            for row in rows
            if row.get("success_rate", 0.0) >= args.min_success_rate
        ]
    else:
        successful = list(rows)
    if not successful:
        raise RuntimeError(
            f"No checkpoint reached success_rate >= {args.min_success_rate:.3f}"
        )

    best_tracking = {
        metric: min(row[metric] for row in successful)
        for metric in TRACKING_METRICS
    }
    eligible = [
        row
        for row in successful
        if all(
            row[metric]
            <= best_tracking[metric] * (1.0 + args.tracking_relative_tolerance)
            for metric in TRACKING_METRICS
        )
    ]
    if not eligible:
        raise RuntimeError("Tracking gate unexpectedly rejected every checkpoint")

    # Rank against the full successful run so a narrowly gated set does not
    # conceal how much a candidate improved or regressed globally.
    metric_ranks = {
        metric: _percentile_ranks(successful, metric) for metric in ranking_metrics
    }
    eligible_labels = {row["label"] for row in eligible}
    for row in rows:
        row["eligible"] = row["label"] in eligible_labels
        row["rank_score"] = sum(
            metric_ranks[metric][row["label"]] for metric in ranking_metrics
        ) / len(ranking_metrics)

    selected = min(eligible, key=lambda row: (row["rank_score"], row["epoch"]))
    ranking = sorted(rows, key=lambda row: (not row["eligible"], row["rank_score"]))
    report = {
        "quality_json": str(quality_path),
        "selection_method": {
            "success_rate_minimum": args.min_success_rate,
            "tracking_relative_tolerance": args.tracking_relative_tolerance,
            "tracking_gate_metrics": list(TRACKING_METRICS),
            "equal_rank_metrics": list(ranking_metrics),
            "description": (
                "success/tracking gate followed by the mean scale-free percentile "
                "rank across tracking and physical-quality metrics"
            ),
        },
        "best_tracking": best_tracking,
        "selected": selected,
        "ranking": ranking,
    }
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label", "epoch", "eligible", "rank_score", "success_rate",
            "evaluator_quality_score", *ranking_metrics, "checkpoint",
        ]
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ranking)

    if args.deploy_checkpoint is not None:
        deploy_checkpoint = args.deploy_checkpoint.expanduser().resolve()
        deploy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected["checkpoint"], deploy_checkpoint)
        report["deploy_checkpoint"] = str(deploy_checkpoint)
        output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        f"Selected {selected['label']} (rank_score={selected['rank_score']:.6f}) "
        f"from {len(eligible)}/{len(rows)} tracking-eligible checkpoints"
    )
    print(f"Checkpoint: {selected['checkpoint']}")


if __name__ == "__main__":
    main()
