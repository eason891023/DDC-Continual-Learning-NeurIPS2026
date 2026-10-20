#!/usr/bin/env python3
"""
Re-evaluate saved continual-learning stage checkpoints with a fixed probe.

This corrects the legacy post-hoc stage probe in ``cl_phase_diagram.py``:
dynamic neuron/synapse state is rebuilt before every individual test sample.
It reports the stage-wise selective-vote readout and the drift of each class's
mean population response from the checkpoint at which that class was learned.

Only compact metrics and class centroids are written; no new dense checkpoint
or per-sample representation files are created.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torchvision import datasets, transforms

from experiments.cl_phase_diagram import apply_complete_model_state, build_fresh_model
from experiments.cl_sigma_sweep import _build_readout_variants


RUN_RE = re.compile(
    r"(?P<condition>distance_sigma(?P<sigma>[0-9.]+)|random)"
    r"_order(?P<order>[0-9]+)_seed(?P<seed>[0-9]+)$"
)
METHODS = ("selective_all10", "selective_seen")


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
        default=Path("outputs/fixed_probe_audit"),
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["distance_sigma0.4", "distance_sigma0.7", "random"],
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[43, 44, 46])
    parser.add_argument("--samples_per_class", type=int, default=20)
    parser.add_argument("--T_on", type=int, default=1000)
    parser.add_argument("--T_off", type=int, default=2000)
    parser.add_argument("--T_skip", type=int, default=100)
    parser.add_argument("--top_k_neurons", type=int, default=100)
    parser.add_argument("--target_mean", type=float, default=0.05)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    return parser.parse_args()


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
        if len(result.get("stage_state_history", [])) != 10:
            continue
        if not all(
            (run_dir / record["stage_state_path"]).exists()
            for record in result["stage_state_history"]
        ):
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "run_name": run_dir.name,
                "condition": condition,
                "seed": seed,
                "result": result,
            }
        )
    if not runs:
        raise FileNotFoundError("No complete saved stage-checkpoint runs matched.")
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


def cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.dot(a, b) / denominator)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, a.size, b.size)
    a_idx = set(np.argpartition(a, -k)[-k:].tolist())
    b_idx = set(np.argpartition(b, -k)[-k:].tolist())
    union = a_idx | b_idx
    return len(a_idx & b_idx) / len(union) if union else float("nan")


def evaluate_one_sample(
    model,
    x: torch.Tensor,
    *,
    T_on: int,
    T_off: int,
    T_skip: int,
) -> torch.Tensor:
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
    return counts


def per_class_accuracy(
    correct: dict[str, dict[int, int]],
    totals: dict[int, int],
    method: str,
) -> dict[str, float]:
    return {
        str(cls): correct[method].get(cls, 0) / total
        for cls, total in sorted(totals.items())
        if total
    }


def bwt_and_forgetting(
    matrix: np.ndarray, class_order: list[int]
) -> tuple[dict[str, float], dict[str, float], float, float]:
    per_bwt: dict[str, float] = {}
    per_forgetting: dict[str, float] = {}
    valid_bwt: list[float] = []
    valid_forgetting: list[float] = []
    for learn_stage, cls in enumerate(class_order):
        final = matrix[len(class_order) - 1, cls]
        learned = matrix[learn_stage, cls]
        history = matrix[learn_stage:, cls]
        if np.isfinite(final) and np.isfinite(learned):
            bwt = float(final - learned)
            forgetting = float(np.nanmax(history) - final)
            per_bwt[str(cls)] = bwt
            per_forgetting[str(cls)] = forgetting
            if learn_stage < len(class_order) - 1:
                valid_bwt.append(bwt)
                valid_forgetting.append(forgetting)
    return (
        per_bwt,
        per_forgetting,
        float(np.mean(valid_bwt)) if valid_bwt else float("nan"),
        float(np.mean(valid_forgetting)) if valid_forgetting else float("nan"),
    )


def run_output_path(args: argparse.Namespace, run: dict) -> Path:
    return args.save_dir / "runs" / f"{run['run_name']}.json"


def evaluate_run(
    run: dict,
    args: argparse.Namespace,
    mnist_test,
    test_picked: dict[int, list[int]],
) -> dict:
    result = run["result"]
    class_order = [int(value) for value in result["class_order"]]
    mode = str(result["mode"])
    sigma = float(result["sigma"])
    seed = int(result["seed"])

    model_args = argparse.Namespace(
        input_mode=result.get("input_mode", "spatial"),
        target_mean=args.target_mean,
    )
    model = build_fresh_model(model_args, mode, sigma, seed, args.device)
    sample, _ = mnist_test[test_picked[class_order[0]][0]]
    model.build(sample.unsqueeze(0).to(args.device))

    matrices = {
        method: np.full((10, 10), np.nan, dtype=np.float32)
        for method in METHODS
    }
    centroids: dict[int, dict[int, np.ndarray]] = {}
    stage_records: list[dict] = []

    for task_idx, cls in enumerate(class_order):
        stage_info = result["stage_state_history"][task_idx]
        state_path = run["run_dir"] / stage_info["stage_state_path"]
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        apply_complete_model_state(model, state, args.device)
        neurons_labels = state["neurons_labels"].to(args.device)
        model.disable_plasticity()
        model.eval()

        seen_classes = class_order[: task_idx + 1]
        seen_tensor = torch.tensor(seen_classes, device=args.device, dtype=torch.long)
        selective = _build_readout_variants(neurons_labels)["selective"][0]

        correct = {method: defaultdict(int) for method in METHODS}
        totals: dict[int, int] = defaultdict(int)
        centroid_sums = {
            seen_cls: torch.zeros(model.exc_neurons, device=args.device)
            for seen_cls in seen_classes
        }

        for true_cls in seen_classes:
            indices = test_picked[true_cls][: args.samples_per_class]
            for dataset_index in indices:
                image, label = mnist_test[dataset_index]
                if int(label) != int(true_cls):
                    raise RuntimeError("Persisted fixed-probe index has wrong label.")
                x = image.unsqueeze(0).to(args.device)
                counts = evaluate_one_sample(
                    model,
                    x,
                    T_on=args.T_on,
                    T_off=args.T_off,
                    T_skip=args.T_skip,
                )
                centroid_sums[true_cls] += counts
                totals[true_cls] += 1

                selective_scores = torch.matmul(counts, selective)
                pred_all = int(selective_scores.argmax().item())
                pred_seen = int(
                    seen_tensor[selective_scores[seen_tensor].argmax()].item()
                )

                correct["selective_all10"][true_cls] += int(pred_all == true_cls)
                correct["selective_seen"][true_cls] += int(pred_seen == true_cls)

        centroids[task_idx] = {
            true_cls: (
                centroid_sums[true_cls] / max(1, totals[true_cls])
            ).detach().cpu().numpy()
            for true_cls in seen_classes
        }
        stage_record = {
            "task_idx": task_idx,
            "class_learned": cls,
            "seen_classes": seen_classes,
            "n_probe_samples": int(sum(totals.values())),
            "per_class": {},
        }
        for method in METHODS:
            per_class = per_class_accuracy(correct, totals, method)
            stage_record[f"accuracy_{method}"] = float(
                sum(correct[method].values()) / max(1, sum(totals.values()))
            )
            stage_record["per_class"][method] = per_class
            for true_cls, accuracy in per_class.items():
                matrices[method][task_idx, int(true_cls)] = accuracy
        stage_records.append(stage_record)
        del state, neurons_labels, selective
        gc.collect()

    drift_records: list[dict] = []
    learn_stage_by_class = {cls: idx for idx, cls in enumerate(class_order)}
    for cls in class_order:
        learn_stage = learn_stage_by_class[cls]
        reference = centroids[learn_stage][cls]
        for task_idx in range(learn_stage, len(class_order)):
            current = centroids[task_idx][cls]
            drift_records.append(
                {
                    "class": cls,
                    "task_idx": task_idx,
                    "tasks_since_learning": task_idx - learn_stage,
                    "cosine_drift": cosine_drift(reference, current),
                    "top_neuron_jaccard": topk_jaccard(
                        reference, current, args.top_k_neurons
                    ),
                }
            )

    method_summaries = {}
    for method, matrix in matrices.items():
        per_bwt, per_forgetting, mean_bwt, mean_forgetting = bwt_and_forgetting(
            matrix, class_order
        )
        method_summaries[method] = {
            "accuracy_matrix": matrix.tolist(),
            "per_class_bwt": per_bwt,
            "per_class_forgetting": per_forgetting,
            "mean_bwt_excluding_last": mean_bwt,
            "mean_forgetting_excluding_last": mean_forgetting,
            "final_average_accuracy": float(np.nanmean(matrix[-1])),
        }

    class0_records = [record for record in drift_records if record["class"] == 0]
    output = {
        "run_name": run["run_name"],
        "condition": run["condition"],
        "mode": mode,
        "sigma": sigma,
        "seed": seed,
        "class_order": class_order,
        "protocol": {
            "samples_per_class": args.samples_per_class,
            "probe_indices_source": str(args.source_dir / "meta.json"),
            "sample_state_policy": "reset_before_every_sample",
            "plasticity_during_probe": False,
            "T_on": args.T_on,
            "T_off": args.T_off,
            "T_skip": args.T_skip,
        },
        "method_summaries": method_summaries,
        "stage_records": stage_records,
        "drift_records": drift_records,
        "class0_drift": class0_records,
    }
    del model
    gc.collect()
    if args.device != "cpu":
        torch.cuda.empty_cache()
    return output


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def aggregate_runs(runs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    aggregate: dict[str, dict] = {}
    for condition, records in sorted(grouped.items()):
        aggregate[condition] = {"n_runs": len(records), "methods": {}}
        for method in METHODS:
            aggregate[condition]["methods"][method] = {}
            for metric in (
                "final_average_accuracy",
                "mean_bwt_excluding_last",
                "mean_forgetting_excluding_last",
            ):
                values = [
                    record["method_summaries"][method][metric] for record in records
                ]
                mean, sd = mean_sd(values)
                aggregate[condition]["methods"][method][metric] = {
                    "mean": mean,
                    "sd": sd,
                    "values": values,
                }
        final_class0_drift = [
            record["class0_drift"][-1]["cosine_drift"] for record in records
        ]
        drift_mean, drift_sd = mean_sd(final_class0_drift)
        aggregate[condition]["class0_final_cosine_drift"] = {
            "mean": drift_mean,
            "sd": drift_sd,
            "values": final_class0_drift,
        }
    return aggregate


def write_report(path: Path, aggregate: dict) -> None:
    lines = [
        "# Sample-Reset Fixed-Probe Forgetting Audit",
        "",
        "Every probe sample starts from rebuilt neutral dynamic state. Synaptic "
        "plasticity is disabled.",
        "",
        "| Condition | N | Readout | Final acc. | Mean BWT | Mean forgetting |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for condition, values in aggregate.items():
        for method, metrics in values["methods"].items():
            lines.append(
                f"| {condition} | {values['n_runs']} | {method} | "
                f"{metrics['final_average_accuracy']['mean']:.3f}±"
                f"{metrics['final_average_accuracy']['sd']:.3f} | "
                f"{metrics['mean_bwt_excluding_last']['mean']:.3f}±"
                f"{metrics['mean_bwt_excluding_last']['sd']:.3f} | "
                f"{metrics['mean_forgetting_excluding_last']['mean']:.3f}±"
                f"{metrics['mean_forgetting_excluding_last']['sd']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Class-0 drift",
            "",
            "| Condition | Final cosine drift from learned checkpoint |",
            "|---|---:|",
        ]
    )
    for condition, values in aggregate.items():
        drift = values["class0_final_cosine_drift"]
        lines.append(f"| {condition} | {drift['mean']:.3f}±{drift['sd']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "runs").mkdir(parents=True, exist_ok=True)
    source_meta = json.loads((args.source_dir / "meta.json").read_text())
    if args.samples_per_class > int(source_meta["per_class_test"]):
        raise ValueError("Requested more probe samples than persisted fixed indices.")
    test_picked = {
        int(cls): [int(index) for index in indices]
        for cls, indices in source_meta["test_picked"].items()
    }
    mnist_test = datasets.MNIST(
        root=str(args.data_root),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    completed: list[dict] = []
    for run in discover_runs(args):
        output_path = run_output_path(args, run)
        if args.resume and output_path.exists():
            print(f"[resume] {run['run_name']}", flush=True)
            completed.append(json.loads(output_path.read_text()))
            continue
        print(f"[evaluate] {run['run_name']}", flush=True)
        output = evaluate_run(run, args, mnist_test, test_picked)
        output_path.write_text(json.dumps(output, indent=2))
        completed.append(output)
        print(f"[saved] {output_path}", flush=True)

    aggregate = aggregate_runs(completed)
    payload = {
        "protocol": {
            "source_dir": str(args.source_dir),
            "source_test_indices_hash": source_meta.get("test_indices_hash"),
            "samples_per_class": args.samples_per_class,
            "sample_state_policy": "reset_before_every_sample",
            "full_dense_outputs_saved": False,
        },
        "aggregate": aggregate,
        "completed_runs": [run["run_name"] for run in completed],
    }
    (args.save_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    write_report(args.save_dir / "REPORT.md", aggregate)
    with (args.save_dir / "run_metrics.csv").open("w", newline="") as handle:
        fieldnames = [
            "run_name",
            "condition",
            "seed",
            "method",
            "final_average_accuracy",
            "mean_bwt_excluding_last",
            "mean_forgetting_excluding_last",
            "class0_final_cosine_drift",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in completed:
            for method in METHODS:
                metrics = run["method_summaries"][method]
                writer.writerow(
                    {
                        "run_name": run["run_name"],
                        "condition": run["condition"],
                        "seed": run["seed"],
                        "method": method,
                        "final_average_accuracy": metrics[
                            "final_average_accuracy"
                        ],
                        "mean_bwt_excluding_last": metrics[
                            "mean_bwt_excluding_last"
                        ],
                        "mean_forgetting_excluding_last": metrics[
                            "mean_forgetting_excluding_last"
                        ],
                        "class0_final_cosine_drift": run["class0_drift"][-1][
                            "cosine_drift"
                        ],
                    }
                )
    print(f"[complete] {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
