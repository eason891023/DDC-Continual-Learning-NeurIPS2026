#!/usr/bin/env python3
"""Aggregate compatible refit-decoder result directories into one report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.evaluate_refit_linear_decoder import aggregate_runs, write_report


PROTOCOL_KEYS = (
    "conditions",
    "calibration_per_class",
    "evaluation_per_class",
    "calibration_indices",
    "evaluation_indices",
    "sample_state_policy",
    "plasticity_during_probe",
    "decoder",
    "decoder_C",
    "decoder_max_iter",
    "decoder_seed",
    "T_on",
    "T_off",
    "T_skip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["distance_sigma0.4", "distance_sigma0.7", "random"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocols = [
        json.loads((input_dir / "protocol.json").read_text())
        for input_dir in args.input_dirs
    ]
    reference = protocols[0]
    for input_dir, protocol in zip(args.input_dirs[1:], protocols[1:]):
        for key in PROTOCOL_KEYS:
            if protocol.get(key) != reference.get(key):
                raise ValueError(
                    f"Protocol mismatch for {key!r} in {input_dir}: "
                    f"{protocol.get(key)!r} != {reference.get(key)!r}"
                )

    records: list[dict] = []
    seen: set[tuple[str, int]] = set()
    allowed_conditions = set(args.conditions)
    allowed_seeds = set(args.seeds)
    for input_dir in args.input_dirs:
        for path in sorted((input_dir / "runs").glob("*.json")):
            record = json.loads(path.read_text())
            key = (str(record["condition"]), int(record["seed"]))
            if key[0] not in allowed_conditions or key[1] not in allowed_seeds:
                continue
            if key in seen:
                raise ValueError(f"Duplicate condition/seed result: {key}")
            seen.add(key)
            records.append(record)

    expected = {
        (condition, seed)
        for condition in args.conditions
        for seed in args.seeds
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise FileNotFoundError(f"Incomplete aggregation: missing={missing}, extra={extra}")

    condition_rank = {value: index for index, value in enumerate(args.conditions)}
    seed_rank = {value: index for index, value in enumerate(args.seeds)}
    records.sort(
        key=lambda record: (
            condition_rank[str(record["condition"])],
            seed_rank[int(record["seed"])],
        )
    )

    protocol = dict(reference)
    protocol["source_dir"] = " + ".join(str(path) for path in args.input_dirs)
    protocol["conditions"] = list(args.conditions)
    protocol["seeds"] = list(args.seeds)
    aggregate = aggregate_runs(records)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.save_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "aggregate": aggregate,
                "completed_runs": [record["run_name"] for record in records],
            },
            indent=2,
        )
    )
    write_report(args.save_dir / "REPORT.md", aggregate, protocol)

    fieldnames = [
        "run_name",
        "condition",
        "seed",
        "frozen_old_class_accuracy",
        "refit_old_class_accuracy",
        "recoverable_old_class_gap",
        "frozen_all_class_accuracy",
        "refit_all_class_accuracy",
        "recoverable_all_class_gap",
    ]
    with (args.save_dir / "run_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_name": record["run_name"],
                    "condition": record["condition"],
                    "seed": record["seed"],
                    **{
                        metric: record["metrics"][metric]
                        for metric in fieldnames[3:]
                    },
                }
            )
    print(f"[complete] aggregated {len(records)} runs -> {args.save_dir}")


if __name__ == "__main__":
    main()
