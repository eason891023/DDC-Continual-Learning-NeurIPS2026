#!/usr/bin/env python3
"""Build the frozen-vote -> final-vote -> final-LR diagnostic ladder."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_dir", type=Path, required=True)
    parser.add_argument("--feature_dirs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["distance_sigma0.4", "distance_sigma0.7", "random"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def condition_from_name(name: str) -> str:
    if "distance_sigma0.4" in name:
        return "distance_sigma0.4"
    if "distance_sigma0.7" in name:
        return "distance_sigma0.7"
    if name.startswith("random_"):
        return "random"
    raise ValueError(f"Unrecognized run name: {name}")


def final_selective_vote(feature_path: Path) -> tuple[float, float]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    calibration_x = payload["calibration_X"].float()
    calibration_y = payload["calibration_y"].long()
    evaluation_x = payload["evaluation_X"].float()
    evaluation_y = payload["evaluation_y"].long()

    counts = [int((calibration_y == cls).sum()) for cls in range(10)]
    if len(set(counts)) != 1:
        raise ValueError(f"Unbalanced calibration classes in {feature_path}: {counts}")
    response = torch.stack(
        [calibration_x[calibration_y == cls].sum(dim=0) for cls in range(10)],
        dim=1,
    )
    selective = response - response.mean(dim=1, keepdim=True)
    predictions = torch.matmul(evaluation_x, selective).argmax(dim=1)
    old_mask = evaluation_y != 9
    old_accuracy = float(
        (predictions[old_mask] == evaluation_y[old_mask]).float().mean()
    )
    all_accuracy = float((predictions == evaluation_y).float().mean())
    return old_accuracy, all_accuracy


def main() -> None:
    args = parse_args()
    metrics_path = args.aggregate_dir / "run_metrics.csv"
    with metrics_path.open() as handle:
        metrics_rows = list(csv.DictReader(handle))
    metrics = {row["run_name"]: row for row in metrics_rows}

    feature_paths: dict[str, Path] = {}
    for feature_dir in args.feature_dirs:
        for path in feature_dir.glob("*.pt"):
            if path.stem in feature_paths:
                raise ValueError(f"Duplicate feature file for {path.stem}")
            feature_paths[path.stem] = path

    expected = {
        (condition, seed)
        for condition in args.conditions
        for seed in args.seeds
    }
    records: list[dict] = []
    observed: set[tuple[str, int]] = set()
    for run_name, path in sorted(feature_paths.items()):
        condition = condition_from_name(run_name)
        seed = int(run_name.rsplit("seed", 1)[1])
        key = (condition, seed)
        if key not in expected:
            continue
        if run_name not in metrics:
            raise FileNotFoundError(f"No aggregate metrics for {run_name}")
        final_vote_old, final_vote_all = final_selective_vote(path)
        frozen_old = float(metrics[run_name]["frozen_old_class_accuracy"])
        refit_old = float(metrics[run_name]["refit_old_class_accuracy"])
        record = {
            "run_name": run_name,
            "condition": condition,
            "seed": seed,
            "frozen_vote_old_accuracy": frozen_old,
            "final_vote_old_accuracy": final_vote_old,
            "final_lr_old_accuracy": refit_old,
            "same_family_recalibration_gain": final_vote_old - frozen_old,
            "linear_decoder_refit_gain": refit_old - frozen_old,
            "decoder_family_gain": refit_old - final_vote_old,
            "final_vote_all_accuracy": final_vote_all,
        }
        records.append(record)
        observed.add(key)
    if observed != expected:
        raise FileNotFoundError(
            f"Incomplete readout ladder: missing={sorted(expected-observed)}"
        )

    condition_rank = {value: index for index, value in enumerate(args.conditions)}
    seed_rank = {value: index for index, value in enumerate(args.seeds)}
    records.sort(
        key=lambda record: (
            condition_rank[record["condition"]], seed_rank[record["seed"]]
        )
    )

    fieldnames = list(records[0])
    with (args.aggregate_dir / "readout_ladder.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    metric_names = fieldnames[3:]
    summary: dict[str, dict] = {}
    for condition in args.conditions:
        summary[condition] = {"n": len(grouped[condition]), "metrics": {}}
        for metric in metric_names:
            values = [float(record[metric]) for record in grouped[condition]]
            mean, sd = mean_sd(values)
            summary[condition]["metrics"][metric] = {
                "mean": mean,
                "sd": sd,
                "values": values,
            }
    (args.aggregate_dir / "readout_ladder_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    lines = [
        "# Final-State Readout Diagnostic",
        "",
        "All final-state calibration and evaluation responses were collected "
        "with recurrent plasticity disabled and dynamic state reset before "
        "every image. The final selective-vote matrix uses the same row-"
        "centered class-response construction as the submitted frozen vote.",
        "",
        "## Readout accuracies",
        "",
        "| Connectivity | N | Submitted frozen selective-vote readout | "
        "Final recalibrated selective-vote readout | Final refit linear decoder |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in args.conditions:
        entry = summary[condition]
        values = entry["metrics"]

        def formatted(metric: str, signed: bool = False) -> str:
            mean = values[metric]["mean"]
            sd = values[metric]["sd"]
            prefix = "+" if signed and mean >= 0 else ""
            return f"{prefix}{mean:.3f}±{sd:.3f}"

        lines.append(
            f"| {condition} | {entry['n']} | "
            f"{formatted('frozen_vote_old_accuracy')} | "
            f"{formatted('final_vote_old_accuracy')} | "
            f"{formatted('final_lr_old_accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## Gains relative to the submitted frozen readout",
            "",
            "| Connectivity | N | Selective-vote recalibration gain | "
            "Linear-decoder refit gain |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in args.conditions:
        entry = summary[condition]
        values = entry["metrics"]

        def formatted_gain(metric: str) -> str:
            mean = values[metric]["mean"]
            sd = values[metric]["sd"]
            prefix = "+" if mean >= 0 else ""
            return f"{prefix}{mean:.3f}±{sd:.3f}"

        lines.append(
            f"| {condition} | {entry['n']} | "
            f"{formatted_gain('same_family_recalibration_gain')} | "
            f"{formatted_gain('linear_decoder_refit_gain')} |"
        )
    lines.extend(
        [
            "",
            "The frozen-to-final vote contrast holds the readout family fixed, "
            "but it is not labeled pure drift because the submitted frozen "
            "matrix was accumulated during sequential training whereas the "
            "final matrix uses a held-out, plasticity-off calibration pass. "
            "The linear-decoder refit gain is the key contrast: "
            "final refit linear-decoder accuracy minus submitted frozen-readout "
            "accuracy. Both gains are paired within seed before aggregation.",
        ]
    )
    (args.aggregate_dir / "READOUT_LADDER.md").write_text("\n".join(lines) + "\n")
    print(f"[complete] {args.aggregate_dir / 'READOUT_LADDER.md'}")


if __name__ == "__main__":
    main()
