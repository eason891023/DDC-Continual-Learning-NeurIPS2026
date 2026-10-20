#!/usr/bin/env python3
"""
Lightweight cross-dataset continual-learning screen.

The primary new condition is CIFAR-10 converted to one grayscale 28x28 channel
so the submitted RSNN architecture, input dimensionality, plasticity rules,
and spatial mapping remain unchanged. This is a semantic-complexity
generalization test, not a claim that RGB information is preserved.

Dense checkpoints are disabled to protect disk space. Each condition/seed is
resumable from its compact result JSON.
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

from experiments.cl_sigma_sweep import (
    READOUT_METHODS,
    format_order_tag,
    parse_class_order,
    run_single,
    set_seed,
)
from main.Network import Brain_Like_RSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["cifar10_gray28", "mnist", "fmnist"],
        default="cifar10_gray28",
    )
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.4, 0.7])
    parser.add_argument("--no_random", action="store_true")
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--per_class", type=int, default=20)
    parser.add_argument("--test_per_class", type=int, default=20)
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
        default=Path("outputs/scope_screens/cifar10_gray28"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    return parser.parse_args()


def load_dataset(name: str, root: str = "data"):
    if name == "cifar10_gray28":
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        )
        train = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        test = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )
        preprocessing = "Grayscale(1) -> Resize(28,28) -> ToTensor"
    elif name == "fmnist":
        transform = transforms.ToTensor()
        train = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform
        )
        test = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform
        )
        preprocessing = "ToTensor"
    else:
        transform = transforms.ToTensor()
        train = datasets.MNIST(
            root=root, train=True, download=True, transform=transform
        )
        test = datasets.MNIST(
            root=root, train=False, download=True, transform=transform
        )
        preprocessing = "ToTensor"
    return train, test, preprocessing


def conditions(args: argparse.Namespace) -> list[tuple[str, float, str]]:
    output = [
        ("distance", sigma, f"distance_sigma{sigma}") for sigma in args.sigmas
    ]
    if not args.no_random:
        output.append(("random", 1.0, "random"))
    return output


def prewarm_mask_cache(args: argparse.Namespace, mode: str, sigma: float, seed: int):
    """Generate/load every mask before the seeded measurement construction."""
    set_seed(seed)
    model = Brain_Like_RSNN(
        inputs_size=torch.Size([1, 1, 28, 28]),
        exc_neurons=4096,
        inh_neurons=1024,
        sigma=sigma,
        dt=0.1,
        mode=mode,
        input_mode="spatial",
        target_mean=args.target_mean,
        seed=seed,
        device=args.device,
    ).to(args.device)
    del model
    gc.collect()
    if args.device != "cpu":
        torch.cuda.empty_cache()


def result_path(save_dir: Path, condition: str, order_tag: str, seed: int) -> Path:
    return save_dir / "runs" / f"{condition}_order{order_tag}_seed{seed}" / "result.json"


def aggregate(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    output: dict[str, dict] = {}
    for condition, rows in sorted(grouped.items()):
        output[condition] = {"n_runs": len(rows), "readouts": {}}
        for method in READOUT_METHODS:
            values = np.asarray(
                [row[f"final_acc_{method}"] for row in rows], dtype=float
            )
            output[condition]["readouts"][method] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return output


def write_report(path: Path, args: argparse.Namespace, summary: dict) -> None:
    lines = [
        f"# Generalization Screen: {args.dataset}",
        "",
        "This stage preserves the submitted RSNN and plasticity. CIFAR-10 is "
        "converted to grayscale 28x28 solely to match the fixed input layer.",
        "",
        "| Condition | N | Selective accuracy |",
        "|---|---:|---:|",
    ]
    for condition, values in summary.items():
        selective = values["readouts"]["selective"]
        lines.append(
            f"| {condition} | {values['n_runs']} | "
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

    train_dataset, test_dataset, preprocessing = load_dataset(args.dataset, str(args.data_root))
    requested_conditions = conditions(args)
    requested_seeds = list(range(args.seed, args.seed + args.n_seeds))

    manifest = {
        "dataset": args.dataset,
        "preprocessing": preprocessing,
        "conditions": [
            {"mode": mode, "sigma": sigma, "condition": condition}
            for mode, sigma, condition in requested_conditions
        ],
        "seeds": requested_seeds,
        "per_class": args.per_class,
        "test_per_class": args.test_per_class,
        "class_order": args.class_order,
        "evaluation_state_policy": args.eval_state_policy,
        "full_checkpoints_saved": False,

    }
    (args.save_dir / "design_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # Pre-generate all cached masks first. The measured construction then
    # starts from a reset RNG and only loads masks, preventing cache-hit/miss
    # differences from changing downstream initial weights.
    for seed in requested_seeds:
        for mode, sigma, _ in requested_conditions:
            print(f"[prewarm] {mode} sigma={sigma} seed={seed}", flush=True)
            prewarm_mask_cache(args, mode, sigma, seed)

    records: list[dict] = []
    for seed in requested_seeds:
        for mode, sigma, condition in requested_conditions:
            path = result_path(args.save_dir, condition, args.order_tag, seed)
            if args.resume and path.exists():
                print(f"[resume] {path}", flush=True)
                records.append(json.loads(path.read_text()))
                continue
            print(f"[run] {condition} seed={seed}", flush=True)
            result = run_single(
                args,
                mode,
                sigma,
                seed,
                train_dataset,
                test_dataset,
                class_order=args.class_order,
                order_tag=args.order_tag,
            )
            path = result_path(
                args.save_dir, result["condition"], result["order_tag"], seed
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2))
            records.append(result)

    summary = aggregate(records)
    (args.save_dir / "aggregate.json").write_text(
        json.dumps({"manifest": manifest, "aggregate": summary}, indent=2)
    )
    write_report(args.save_dir / "REPORT.md", args, summary)
    print(f"[complete] {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
