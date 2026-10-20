#!/usr/bin/env python3
"""
Post-hoc final-state linear-decoder diagnostic requested during review.

For each saved continual-learning run, this script restores the final recurrent
state, disables plasticity, and rebuilds neutral dynamic state before every
image.  It collects excitatory spike-count vectors on two disjoint subsets of
the held-out MNIST test pool:

* decoder calibration: examples never used to train the recurrent network;
* decoder evaluation: the same first examples/class used by the fixed-probe
  audit, and never used to fit the decoder.

A ten-way L2 logistic decoder is fit only on final-state calibration vectors.
Its accuracy on old classes is reported alongside the submitted frozen
selective-vote observer on the identical evaluation vectors.  The diagnostic
does not retrain or modify the recurrent network.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

from experiments.cl_phase_diagram import apply_complete_model_state, build_fresh_model
from experiments.cl_sigma_sweep import _build_readout_variants


RUN_RE = re.compile(
    r"(?P<condition>distance_sigma(?P<sigma>[0-9.]+)|random)"
    r"_order(?P<order>[0-9]+)_seed(?P<seed>[0-9]+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("outputs/phase_diagram"),
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("outputs/readout_ladder"),
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["distance_sigma0.4", "distance_sigma0.7", "random"],
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--calibration_per_class", type=int, default=30)
    parser.add_argument("--evaluation_per_class", type=int, default=20)
    parser.add_argument("--T_on", type=int, default=1000)
    parser.add_argument("--T_off", type=int, default=2000)
    parser.add_argument("--T_skip", type=int, default=100)
    parser.add_argument("--target_mean", type=float, default=0.05)
    parser.add_argument("--decoder_C", type=float, default=1.0)
    parser.add_argument("--decoder_max_iter", type=int, default=3000)
    parser.add_argument("--decoder_seed", type=int, default=20260731)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--save_features", action="store_true")
    return parser.parse_args()


def index_hash(indices: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in indices).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), sd


def discover_runs(args: argparse.Namespace) -> list[dict]:
    allowed_conditions = set(args.conditions)
    allowed_seeds = set(args.seeds)
    runs: list[dict] = []
    for run_dir in sorted((args.source_dir / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        condition = match.group("condition")
        seed = int(match.group("seed"))
        result_path = run_dir / "result.json"
        if (
            condition not in allowed_conditions
            or seed not in allowed_seeds
            or not result_path.exists()
        ):
            continue
        result = json.loads(result_path.read_text())
        state_path = run_dir / result["trained_complete_state_path"]
        if not state_path.exists():
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "run_name": run_dir.name,
                "condition": condition,
                "seed": seed,
                "result": result,
                "state_path": state_path,
            }
        )
    expected = len(args.conditions) * len(args.seeds)
    if len(runs) != expected:
        raise FileNotFoundError(
            f"Expected {expected} complete runs, found {len(runs)}."
        )
    condition_rank = {
        condition: rank for rank, condition in enumerate(args.conditions)
    }
    seed_rank = {seed: rank for rank, seed in enumerate(args.seeds)}
    runs.sort(
        key=lambda run: (
            condition_rank[run["condition"]],
            seed_rank[run["seed"]],
        )
    )
    return runs


def build_splits(
    source_meta: dict,
    *,
    evaluation_per_class: int,
    calibration_per_class: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    evaluation: dict[int, list[int]] = {}
    calibration: dict[int, list[int]] = {}
    required = evaluation_per_class + calibration_per_class
    for cls_string, indices in sorted(
        source_meta["test_picked"].items(), key=lambda item: int(item[0])
    ):
        cls = int(cls_string)
        values = [int(value) for value in indices]
        if len(values) < required:
            raise ValueError(
                f"Class {cls} has {len(values)} test examples; need {required}."
            )
        evaluation[cls] = values[:evaluation_per_class]
        calibration[cls] = values[
            evaluation_per_class : evaluation_per_class + calibration_per_class
        ]
        if set(evaluation[cls]) & set(calibration[cls]):
            raise RuntimeError(f"Calibration/evaluation overlap for class {cls}.")
    return calibration, evaluation


def collect_one(
    model,
    image: torch.Tensor,
    *,
    T_on: int,
    T_off: int,
    T_skip: int,
) -> torch.Tensor:
    x = image.unsqueeze(0).to(model.device)
    model.build(x)
    counts = torch.zeros(model.exc_neurons, device=x.device)
    with torch.no_grad():
        for step in range(T_on):
            exc_spikes, _ = model(x)
            if step >= T_skip:
                counts += exc_spikes[0]
            model.step()
        zero_input = torch.zeros_like(x)
        for _ in range(T_off):
            exc_spikes, _ = model(zero_input)
            counts += exc_spikes[0]
            model.step()
    return counts.detach().cpu()


def collect_split(
    model,
    dataset,
    indices_by_class: dict[int, list[int]],
    *,
    T_on: int,
    T_off: int,
    T_skip: int,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[int] = []
    dataset_indices: list[int] = []
    total = sum(len(values) for values in indices_by_class.values())
    completed = 0
    for cls, indices in sorted(indices_by_class.items()):
        for dataset_index in indices:
            image, label = dataset[dataset_index]
            if int(label) != cls:
                raise RuntimeError(
                    f"{split_name} index {dataset_index}: expected {cls}, got {label}."
                )
            features.append(
                collect_one(
                    model,
                    image,
                    T_on=T_on,
                    T_off=T_off,
                    T_skip=T_skip,
                )
            )
            labels.append(cls)
            dataset_indices.append(dataset_index)
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(
                    f"  [{split_name}] {completed}/{total}",
                    flush=True,
                )
    return (
        torch.stack(features),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(dataset_indices, dtype=torch.long),
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_true[mask] == y_pred[mask]))


def per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        str(int(cls)): float(np.mean(y_pred[y_true == cls] == cls))
        for cls in sorted(np.unique(y_true))
    }


def run_output_path(args: argparse.Namespace, run: dict) -> Path:
    return args.save_dir / "runs" / f"{run['run_name']}.json"


def evaluate_run(
    run: dict,
    args: argparse.Namespace,
    mnist_test,
    calibration_indices: dict[int, list[int]],
    evaluation_indices: dict[int, list[int]],
) -> dict:
    result = run["result"]
    class_order = [int(value) for value in result["class_order"]]
    old_classes = class_order[:-1]
    mode = str(result["mode"])
    sigma = float(result["sigma"])
    seed = int(result["seed"])

    model_args = argparse.Namespace(
        input_mode=result.get("input_mode", "spatial"),
        target_mean=args.target_mean,
    )
    model = build_fresh_model(model_args, mode, sigma, seed, args.device)
    sample, _ = mnist_test[evaluation_indices[class_order[0]][0]]
    model.build(sample.unsqueeze(0).to(args.device))
    state = torch.load(run["state_path"], map_location="cpu", weights_only=False)
    apply_complete_model_state(model, state, args.device)
    neurons_labels = state["neurons_labels"].to(args.device)
    model.disable_plasticity()
    model.eval()

    print(f"[run] {run['run_name']}", flush=True)
    calibration_X, calibration_y, calibration_idx = collect_split(
        model,
        mnist_test,
        calibration_indices,
        T_on=args.T_on,
        T_off=args.T_off,
        T_skip=args.T_skip,
        split_name="calibration",
    )
    evaluation_X, evaluation_y, evaluation_idx = collect_split(
        model,
        mnist_test,
        evaluation_indices,
        T_on=args.T_on,
        T_off=args.T_off,
        T_skip=args.T_skip,
        split_name="evaluation",
    )

    selective = _build_readout_variants(neurons_labels)["selective"][0]
    frozen_predictions = torch.matmul(
        evaluation_X.to(args.device), selective
    ).argmax(dim=1).cpu().numpy()

    decoder = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.decoder_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=args.decoder_max_iter,
            random_state=args.decoder_seed,
        ),
    )
    decoder.fit(calibration_X.numpy(), calibration_y.numpy())
    refit_predictions = decoder.predict(evaluation_X.numpy())

    y_eval = evaluation_y.numpy()
    old_mask = np.isin(y_eval, np.asarray(old_classes))
    all_mask = np.ones_like(y_eval, dtype=bool)
    frozen_old = accuracy(y_eval, frozen_predictions, old_mask)
    refit_old = accuracy(y_eval, refit_predictions, old_mask)
    frozen_all = accuracy(y_eval, frozen_predictions, all_mask)
    refit_all = accuracy(y_eval, refit_predictions, all_mask)

    output = {
        "run_name": run["run_name"],
        "condition": run["condition"],
        "mode": mode,
        "sigma": sigma,
        "seed": seed,
        "class_order": class_order,
        "old_classes": old_classes,
        "protocol": {
            "source_state_path": str(run["state_path"]),
            "plasticity_during_probe": False,
            "sample_state_policy": "reset_before_every_sample",
            "feature": "excitatory_spike_counts",
            "feature_dim": int(evaluation_X.shape[1]),
            "decoder": "StandardScaler + 10-way L2 LogisticRegression",
            "decoder_C": args.decoder_C,
            "decoder_max_iter": args.decoder_max_iter,
            "decoder_seed": args.decoder_seed,
            "calibration_per_class": args.calibration_per_class,
            "evaluation_per_class": args.evaluation_per_class,
            "calibration_indices_hash": index_hash(calibration_idx.tolist()),
            "evaluation_indices_hash": index_hash(evaluation_idx.tolist()),
            "calibration_evaluation_disjoint": not bool(
                set(calibration_idx.tolist()) & set(evaluation_idx.tolist())
            ),
            "T_on": args.T_on,
            "T_off": args.T_off,
            "T_skip": args.T_skip,
        },
        "metrics": {
            "frozen_old_class_accuracy": frozen_old,
            "refit_old_class_accuracy": refit_old,
            "recoverable_old_class_gap": refit_old - frozen_old,
            "frozen_all_class_accuracy": frozen_all,
            "refit_all_class_accuracy": refit_all,
            "recoverable_all_class_gap": refit_all - frozen_all,
            "frozen_per_class_accuracy": per_class_accuracy(
                y_eval, frozen_predictions
            ),
            "refit_per_class_accuracy": per_class_accuracy(
                y_eval, refit_predictions
            ),
        },
        "predictions": {
            "labels": y_eval.tolist(),
            "dataset_indices": evaluation_idx.tolist(),
            "frozen": frozen_predictions.tolist(),
            "refit": refit_predictions.tolist(),
        },
    }

    if args.save_features:
        feature_path = args.save_dir / "features" / f"{run['run_name']}.pt"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "calibration_X": calibration_X,
                "calibration_y": calibration_y,
                "calibration_indices": calibration_idx,
                "evaluation_X": evaluation_X,
                "evaluation_y": evaluation_y,
                "evaluation_indices": evaluation_idx,
            },
            feature_path,
        )
        output["protocol"]["feature_path"] = str(feature_path)

    print(
        f"[metrics] frozen_old={frozen_old:.4f} "
        f"refit_old={refit_old:.4f} gap={refit_old - frozen_old:+.4f}",
        flush=True,
    )
    del state, neurons_labels, selective, model
    gc.collect()
    if args.device != "cpu":
        torch.cuda.empty_cache()
    return output


def aggregate_runs(runs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    aggregate: dict[str, dict] = {}
    metric_names = (
        "frozen_old_class_accuracy",
        "refit_old_class_accuracy",
        "recoverable_old_class_gap",
        "frozen_all_class_accuracy",
        "refit_all_class_accuracy",
        "recoverable_all_class_gap",
    )
    for condition, records in grouped.items():
        aggregate[condition] = {"n_runs": len(records), "metrics": {}}
        for metric in metric_names:
            values = [record["metrics"][metric] for record in records]
            mean, sd = mean_sd(values)
            aggregate[condition]["metrics"][metric] = {
                "mean": mean,
                "sd": sd,
                "values": values,
            }
    return aggregate


def write_report(path: Path, aggregate: dict, protocol: dict) -> None:
    lines = [
        "# Final-State Refit Linear-Decoder Diagnostic",
        "",
        "Plasticity is disabled and dynamic state is reset before every image. "
        "The ten-way L2 linear decoder is calibrated on final-state spike "
        "vectors disjoint from the evaluation vectors. Frozen and refit "
        "decoders are evaluated on the identical old-class examples.",
        "",
        "| Condition | N | Frozen old ACC | Refit old ACC | Refit−frozen |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in protocol["conditions"]:
        values = aggregate[condition]
        metrics = values["metrics"]
        lines.append(
            f"| {condition} | {values['n_runs']} | "
            f"{metrics['frozen_old_class_accuracy']['mean']:.3f}±"
            f"{metrics['frozen_old_class_accuracy']['sd']:.3f} | "
            f"{metrics['refit_old_class_accuracy']['mean']:.3f}±"
            f"{metrics['refit_old_class_accuracy']['sd']:.3f} | "
            f"{metrics['recoverable_old_class_gap']['mean']:+.3f}±"
            f"{metrics['recoverable_old_class_gap']['sd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The refit−frozen gap measures the component of frozen-observer "
            "loss that a post-hoc linear decoder can absorb. It is a diagnostic "
            "of recoverable coordinate drift, not a replacement benchmark.",
            "",
            "## Protocol",
            "",
            f"- Source: `{protocol['source_dir']}`",
            f"- Conditions: `{protocol['conditions']}`",
            f"- Seeds: `{protocol['seeds']}`",
            f"- Calibration/evaluation per class: "
            f"`{protocol['calibration_per_class']}/"
            f"{protocol['evaluation_per_class']}`",
            "- Primary endpoint: old classes (all classes except the final one).",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "runs").mkdir(parents=True, exist_ok=True)
    source_meta = json.loads((args.source_dir / "meta.json").read_text())
    calibration_indices, evaluation_indices = build_splits(
        source_meta,
        evaluation_per_class=args.evaluation_per_class,
        calibration_per_class=args.calibration_per_class,
    )
    runs = discover_runs(args)
    mnist_test = datasets.MNIST(
        root=str(args.data_root),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    protocol = {
        "source_dir": str(args.source_dir),
        "source_test_indices_hash": source_meta.get("test_indices_hash"),
        "conditions": args.conditions,
        "seeds": args.seeds,
        "calibration_per_class": args.calibration_per_class,
        "evaluation_per_class": args.evaluation_per_class,
        "calibration_indices": calibration_indices,
        "evaluation_indices": evaluation_indices,
        "sample_state_policy": "reset_before_every_sample",
        "plasticity_during_probe": False,
        "decoder": "StandardScaler + 10-way L2 LogisticRegression",
        "decoder_C": args.decoder_C,
        "decoder_max_iter": args.decoder_max_iter,
        "decoder_seed": args.decoder_seed,
        "T_on": args.T_on,
        "T_off": args.T_off,
        "T_skip": args.T_skip,
    }
    (args.save_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    completed: list[dict] = []
    for run in runs:
        output_path = run_output_path(args, run)
        if args.resume and output_path.exists():
            print(f"[resume] {run['run_name']}", flush=True)
            completed.append(json.loads(output_path.read_text()))
            continue
        output = evaluate_run(
            run,
            args,
            mnist_test,
            calibration_indices,
            evaluation_indices,
        )
        output_path.write_text(json.dumps(output, indent=2))
        completed.append(output)

    aggregate = aggregate_runs(completed)
    summary = {
        "protocol": protocol,
        "aggregate": aggregate,
        "completed_runs": [run["run_name"] for run in completed],
    }
    (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(args.save_dir / "REPORT.md", aggregate, protocol)

    with (args.save_dir / "run_metrics.csv").open("w", newline="") as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in completed:
            writer.writerow(
                {
                    "run_name": run["run_name"],
                    "condition": run["condition"],
                    "seed": run["seed"],
                    **{
                        metric: run["metrics"][metric]
                        for metric in fieldnames[3:]
                    },
                }
            )
    print(f"[complete] {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
