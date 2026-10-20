#!/usr/bin/env python3
"""
Excitatory-plasticity component robustness screen.

Compares the submitted full rule against ``triplet_only_exc``, which sets the
heterosynaptic and transmitter-induced coefficients to zero in both plastic
excitatory pathways while preserving triplet STDP and the original inhibitory
homeostatic rule. This is a controlled component ablation, not a claim to
implement every possible pair-based STDP variant.

The design crosses plasticity variant with intermediate DDC versus
density-matched random topology. It saves compact results without checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from experiments.cl_generalization_screen import prewarm_mask_cache
from experiments.cl_sigma_sweep import (
    READOUT_METHODS,
    format_order_tag,
    parse_class_order,
    run_single,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--per_class", type=int, default=10)
    parser.add_argument("--test_per_class", type=int, default=10)
    parser.add_argument("--T_on", type=int, default=1000)
    parser.add_argument("--T_off", type=int, default=2000)
    parser.add_argument("--T_skip", type=int, default=100)
    parser.add_argument("--burn_in_on", type=int, default=2000)
    parser.add_argument("--burn_in_off", type=int, default=3000)
    parser.add_argument("--target_mean", type=float, default=0.05)
    parser.add_argument("--class_order", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("outputs/scope_screens/plasticity_triplet_only"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    return parser.parse_args()


def aggregate(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    output: dict[str, dict] = {}
    for condition, rows in sorted(grouped.items()):
        output[condition] = {
            "n_runs": len(rows),
            "mode": rows[0]["mode"],
            "plasticity_variant": rows[0]["plasticity_variant"],
            "readouts": {},
        }
        for method in READOUT_METHODS:
            values = np.asarray(
                [row[f"final_acc_{method}"] for row in rows], dtype=float
            )
            output[condition]["readouts"][method] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "values": values.tolist(),
            }
    return output


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Excitatory Plasticity Component Robustness",
        "",
        "| Topology | Plasticity | N | Selective accuracy |",
        "|---|---|---:|---:|",
    ]
    for values in summary.values():
        selective = values["readouts"]["selective"]
        topology = "DDC" if values["mode"] == "distance" else "random"
        lines.append(
            f"| {topology} | {values['plasticity_variant']} | "
            f"{values['n_runs']} | "
            f"{selective['mean']:.3f}±{selective['sd']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "runs").mkdir(parents=True, exist_ok=True)
    args.class_order = parse_class_order(args.class_order)
    args.order_tag = format_order_tag(args.class_order)
    args.input_mode = "spatial"
    args.eval_batch_size = 1
    args.eval_state_policy = "sample_reset"
    args.interim_test_per_class = args.test_per_class
    args.skip_interim = True
    args.save_task_snapshots = False
    args.task_snapshot_weights = "none"
    args.save_checkpoints = False

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=str(args.data_root), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(args.data_root), train=False, download=True, transform=transform
    )
    modes = ["distance", "random"]
    variants = ["full", "triplet_only_exc"]
    requested_seeds = list(range(args.seed, args.seed + args.n_seeds))
    manifest = {

        "dataset": "mnist",
        "modes": modes,
        "sigma": args.sigma,
        "plasticity_variants": {
            "full": "submitted rule",
            "triplet_only_exc": (
                "beta=0 and delta=0 in input-to-E and recurrent E-to-E LTP; "
                "triplet STDP and inhibitory homeostatic plasticity retained"
            ),
        },
        "seeds": requested_seeds,
        "per_class": args.per_class,
        "test_per_class": args.test_per_class,
        "evaluation_state_policy": "sample_reset",
        "full_checkpoints_saved": False,
    }
    (args.save_dir / "design_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    for seed in requested_seeds:
        for mode in modes:
            print(f"[prewarm] {mode} seed={seed}", flush=True)
            prewarm_mask_cache(args, mode, args.sigma, seed)

    records: list[dict] = []
    for seed in requested_seeds:
        for mode in modes:
            for variant in variants:
                args.plasticity_variant = variant
                args.condition_suffix = variant
                topology = (
                    f"distance_sigma{args.sigma}"
                    if mode == "distance"
                    else "random"
                )
                condition = f"{topology}__{variant}"
                output_path = (
                    args.save_dir / "runs" / f"{condition}_seed{seed}.json"
                )
                if args.resume and output_path.exists():
                    print(f"[resume] {output_path}", flush=True)
                    records.append(json.loads(output_path.read_text()))
                    continue
                print(f"[run] {condition} seed={seed}", flush=True)
                result = run_single(
                    args,
                    mode,
                    args.sigma,
                    seed,
                    train_dataset,
                    test_dataset,
                    class_order=args.class_order,
                    order_tag=args.order_tag,
                )
                output_path.write_text(json.dumps(result, indent=2))
                records.append(result)
                gc.collect()
                if args.device != "cpu":
                    torch.cuda.empty_cache()

    summary = aggregate(records)
    (args.save_dir / "aggregate.json").write_text(
        json.dumps({"manifest": manifest, "aggregate": summary}, indent=2)
    )
    write_report(args.save_dir / "REPORT.md", summary)
    print(f"[complete] {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
