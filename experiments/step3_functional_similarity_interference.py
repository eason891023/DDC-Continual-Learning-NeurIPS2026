"""
Pairwise rewriting assay (paper Section 7, Fig. 5, Fig. A7).

Protocol for each (condition, class pair A -> B, seed):
  1. train class A
  2. snapshot W_A and class-A activity affinity
  3. train class B
  4. snapshot W_B and class-B activity affinity

Important synapses use the same functional definition as the benchmark analyses
(Section 6, Fig. 4):
  E->E     score_ij = |W_A_ij| * affinity_i(A) * affinity_j(A)
  Input->E score_pj = |W_A_pj| * affinity_j(A)

It then reports similar-vs-dissimilar pair summaries for:
  - mean directional interference: mean(|W_B - W_A| on A-important synapses)
  - total directional interference: sum(|W_B - W_A| on A-important synapses)
  - contention: Jaccard(A-important, B-important)

This is a two-class pair assay, not a full 10-class benchmark analysis. The paper
reports the recurrent E->E mean directional interference ("rewriting").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.Network import Brain_Like_RSNN
from experiments.benchmark_interference import (
    compute_ee_importance_indices,
    compute_input_importance_indices,
    flat_delta_lookup,
    jaccard_from_flat_indices,
)
from experiments.step3_synapse_interference import (
    compute_delta,
    compute_interference_metrics,
    get_per_class_subsets,
    set_seed,
    snapshot_weights,
    train_one_class,
)


SIMILAR_PAIRS = [(3, 5), (5, 8), (3, 8)]
DISSIMILAR_PAIRS = [(0, 1), (1, 4), (0, 7)]
CONDITION_COLORS = {
    "sigma0.4": "#4DAF4A",
    "sigma0.7": "#2166AC",
    "random": "#D6604D",
}


def parse_pair(text: str) -> tuple[int, int]:
    parts = text.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected pair like '3,5', got {text!r}")
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer pair, got {text!r}") from exc
    return a, b


def pair_group(cls_a: int, cls_b: int, similar_pairs: Iterable[tuple[int, int]], dissimilar_pairs: Iterable[tuple[int, int]]) -> str:
    pair = tuple(sorted((int(cls_a), int(cls_b))))
    similar = {tuple(sorted(p)) for p in similar_pairs}
    dissimilar = {tuple(sorted(p)) for p in dissimilar_pairs}
    if pair in similar:
        return "similar"
    if pair in dissimilar:
        return "dissimilar"
    return "other"


def neurons_labels_for_class(spike_counts: torch.Tensor, cls: int, n_classes: int = 10) -> torch.Tensor:
    labels = torch.zeros((int(spike_counts.numel()), n_classes), device=spike_counts.device, dtype=spike_counts.dtype)
    labels[:, int(cls)] = spike_counts
    return labels


def layer_functional_metrics(
    layer: str,
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    cls_a: int,
    cls_b: int,
    top_k: int,
) -> dict:
    if layer == "ee":
        flat_a, _ = compute_ee_importance_indices(weights_a, labels_a, cls_a, top_k)
        flat_b, _ = compute_ee_importance_indices(weights_b, labels_b, cls_b, top_k)
    elif layer == "input":
        flat_a, _ = compute_input_importance_indices(weights_a, labels_a, cls_a, top_k)
        flat_b, _ = compute_input_importance_indices(weights_b, labels_b, cls_b, top_k)
    else:
        raise ValueError(f"Unknown layer: {layer}")

    if flat_a.numel() == 0:
        return {
            "topk": 0,
            "mean_interference": float("nan"),
            "median_interference": float("nan"),
            "total_interference": float("nan"),
            "contention_jaccard": float("nan"),
        }

    delta = flat_delta_lookup(weights_a, weights_b, flat_a.to(weights_a.device))
    return {
        "topk": int(flat_a.numel()),
        "mean_interference": float(delta.mean().item()),
        "median_interference": float(delta.median().item()),
        "total_interference": float(delta.sum().item()),
        "contention_jaccard": float(jaccard_from_flat_indices(flat_a, flat_b)),
    }


def compute_functional_interference(
    w_a: dict[str, torch.Tensor],
    w_b: dict[str, torch.Tensor],
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    cls_a: int,
    cls_b: int,
    top_k_ee: int,
    top_k_input: int,
) -> dict:
    input_metrics = layer_functional_metrics(
        "input",
        w_a["input_EE"],
        w_b["input_EE"],
        labels_a,
        labels_b,
        cls_a,
        cls_b,
        top_k_input,
    )
    ee_metrics = layer_functional_metrics(
        "ee",
        w_a["RSNN_EE"],
        w_b["RSNN_EE"],
        labels_a,
        labels_b,
        cls_a,
        cls_b,
        top_k_ee,
    )

    return {
        "input_EE": input_metrics,
        "RSNN_EE": ee_metrics,
        "combined_mean_interference": 0.5 * (
            input_metrics["mean_interference"] + ee_metrics["mean_interference"]
        ),
        "combined_total_interference": input_metrics["total_interference"] + ee_metrics["total_interference"],
        "combined_contention_jaccard": 0.5 * (
            input_metrics["contention_jaccard"] + ee_metrics["contention_jaccard"]
        ),
    }


def build_model(args: argparse.Namespace, mode: str, sigma: float, seed: int, sample_img: torch.Tensor) -> Brain_Like_RSNN:
    set_seed(seed)
    model = Brain_Like_RSNN(
        inputs_size=torch.Size([1, 1, 28, 28]),
        exc_neurons=args.exc_neurons,
        inh_neurons=args.inh_neurons,
        sigma=sigma,
        dt=0.1,
        mode=mode,
        input_mode=args.input_mode,
        target_mean=args.target_mean,
        device=args.device,
    ).to(args.device)
    model.build(sample_img.unsqueeze(0).to(args.device))
    return model


def run_single(
    args: argparse.Namespace,
    mode: str,
    sigma: float,
    seed: int,
    cls_a: int,
    cls_b: int,
    mnist_train,
) -> dict:
    set_seed(seed)
    train_subsets = get_per_class_subsets(mnist_train, per_class=args.per_class, seed=seed)

    model = build_model(args, mode, sigma, seed, train_subsets[0][0][0])

    all_train_indices = []
    for cls in range(10):
        all_train_indices.extend(train_subsets[cls].indices)
    burn_loader = DataLoader(Subset(mnist_train, all_train_indices), batch_size=1, shuffle=False)
    model.burn_in(burn_loader, T_on=args.burn_in_on, T_off=args.burn_in_off)

    mask_dict = {
        "input_EE": model.inputs2RSNN_EE_synapase.inputs_EE_mask,
        "RSNN_EE": model.RSNN_EE_synapase.RSNN_EE_mask,
        "RSNN_IE": model.RSNN_IE_synapase.IE_weights_mask,
    }

    w0 = snapshot_weights(model)

    loader_a = DataLoader(train_subsets[cls_a], batch_size=1, shuffle=False)
    spike_counts_a = train_one_class(model, loader_a, cls_a, args.T_on, args.T_off, args.T_skip, args.device)
    w_a = snapshot_weights(model)
    labels_a = neurons_labels_for_class(spike_counts_a, cls_a)

    loader_b = DataLoader(train_subsets[cls_b], batch_size=1, shuffle=False)
    spike_counts_b = train_one_class(model, loader_b, cls_b, args.T_on, args.T_off, args.T_skip, args.device)
    w_b = snapshot_weights(model)
    labels_b = neurons_labels_for_class(spike_counts_b, cls_b)

    functional = compute_functional_interference(
        w_a,
        w_b,
        labels_a,
        labels_b,
        cls_a,
        cls_b,
        top_k_ee=args.top_k_ee,
        top_k_input=args.top_k_input,
    )

    if args.include_delta_top5_reference:
        delta_a = compute_delta(w_a, w0)
        delta_b = compute_delta(w_b, w_a)
        original = compute_interference_metrics(delta_a, delta_b, mask_dict)
    else:
        original = None

    del model
    if args.device != "cpu":
        torch.cuda.empty_cache()

    return {
        "mode": mode,
        "sigma": sigma,
        "condition": condition_key(mode, sigma),
        "seed": seed,
        "cls_A": int(cls_a),
        "cls_B": int(cls_b),
        "group": pair_group(cls_a, cls_b, args.similar_pairs, args.dissimilar_pairs),
        "functional_metrics": functional,
        "delta_top5_reference_metrics": original,
    }


def condition_key(mode: str, sigma: float) -> str:
    if mode == "random":
        return "random"
    return f"sigma{sigma:g}"


def conditions_from_args(args: argparse.Namespace) -> list[tuple[str, float]]:
    conditions = [("distance", float(sigma)) for sigma in args.sigmas]
    if args.include_random:
        conditions.append(("random", 1.0))
    return conditions


def mean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def std_or_zero(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0


def summarize(records: list[dict]) -> dict:
    metrics = [
        "combined_mean_interference",
        "combined_total_interference",
        "combined_contention_jaccard",
        "input_EE.mean_interference",
        "input_EE.total_interference",
        "input_EE.contention_jaccard",
        "RSNN_EE.mean_interference",
        "RSNN_EE.total_interference",
        "RSNN_EE.contention_jaccard",
    ]

    def get_metric(rec: dict, metric: str) -> float:
        payload = rec["functional_metrics"]
        if "." not in metric:
            return float(payload[metric])
        layer, key = metric.split(".", 1)
        return float(payload[layer][key])

    summary: dict = {}
    for condition in sorted({r["condition"] for r in records}):
        summary[condition] = {}
        for group in ["similar", "dissimilar"]:
            subset = [r for r in records if r["condition"] == condition and r["group"] == group]
            summary[condition][group] = {}
            for metric in metrics:
                values = [get_metric(r, metric) for r in subset]
                summary[condition][group][metric] = {
                    "mean": mean_or_nan(values),
                    "std": std_or_zero(values),
                    "n": int(len(values)),
                    "values": values,
                }

        summary[condition]["tests"] = {}
        for metric in metrics:
            sim = summary[condition]["similar"][metric]["values"]
            dis = summary[condition]["dissimilar"][metric]["values"]
            sim_arr = np.asarray(sim, dtype=float)
            dis_arr = np.asarray(dis, dtype=float)
            if sim_arr.size and dis_arr.size:
                t_stat, t_p = stats.ttest_ind(sim_arr, dis_arr, equal_var=False, nan_policy="omit")
                u_stat, u_p = stats.mannwhitneyu(sim_arr, dis_arr, alternative="two-sided")
                ratio = float(np.nanmean(sim_arr) / (np.nanmean(dis_arr) + 1e-12))
            else:
                t_stat = t_p = u_stat = u_p = ratio = float("nan")
            summary[condition]["tests"][metric] = {
                "similar_over_dissimilar_ratio": ratio,
                "t_stat": float(t_stat),
                "t_p": float(t_p),
                "mann_whitney_u": float(u_stat),
                "mann_whitney_p": float(u_p),
            }
    return summary


def condition_sort_key(condition: str) -> tuple[int, float]:
    if condition == "random":
        return (1, float("inf"))
    return (0, float(condition.replace("sigma", "")))


def plot_summary(summary: dict, out_path: Path) -> None:
    conditions = sorted(summary.keys(), key=condition_sort_key)
    groups = ["similar", "dissimilar"]
    labels = []
    colors = []
    mean_vals = []
    mean_errs = []
    total_vals = []
    total_errs = []

    for condition in conditions:
        for group in groups:
            labels.append(f"{condition.replace('sigma', 'sigma=')}\n{group}")
            colors.append(CONDITION_COLORS.get(condition, "#777777"))
            mean_rec = summary[condition][group]["combined_mean_interference"]
            total_rec = summary[condition][group]["combined_total_interference"]
            mean_vals.append(mean_rec["mean"])
            mean_errs.append(mean_rec["std"])
            total_vals.append(total_rec["mean"])
            total_errs.append(total_rec["std"])

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    for ax, vals, errs, ylabel, title in [
        (
            axes[0],
            mean_vals,
            mean_errs,
            "Mean directional interference\n0.5 x (Input->E + E->E)",
            "(a) Mean directional interference",
        ),
        (
            axes[1],
            total_vals,
            total_errs,
            "Total directional interference\nInput->E + E->E",
            "(b) Total rewriting on functional A-important synapses",
        ),
    ]:
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, edgecolor="black", linewidth=0.7)
        for bar, label in zip(bars, labels):
            if "dissimilar" in label:
                bar.set_hatch("//")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="#D9D9D9", alpha=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(args.similar_pairs) + list(args.dissimilar_pairs)
    conditions = conditions_from_args(args)
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    all_results: list[dict] = []
    total_runs = len(conditions) * len(pairs) * len(seeds)

    run_idx = 0
    for mode, sigma in conditions:
        for cls_a, cls_b in pairs:
            for seed in seeds:
                run_idx += 1
                condition = condition_key(mode, sigma)
                group = pair_group(cls_a, cls_b, args.similar_pairs, args.dissimilar_pairs)
                print(f"\n[{run_idx}/{total_runs}] {condition} | {group} pair ({cls_a},{cls_b}) | seed {seed}")
                record = run_single(args, mode, sigma, seed, cls_a, cls_b, mnist_train)
                all_results.append(record)
                (save_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))

    summary = summarize(all_results)
    payload = {
        "description": "Pairwise rewriting assay using functional important synapses.",
        "similar_pairs": [list(p) for p in args.similar_pairs],
        "dissimilar_pairs": [list(p) for p in args.dissimilar_pairs],
        "top_k_ee": args.top_k_ee,
        "top_k_input": args.top_k_input,
        "records": all_results,
        "summary": summary,
    }
    (save_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    plot_summary(summary, save_dir / "functional_similarity_interference_summary.png")

    print(f"\nSaved records -> {save_dir / 'all_results.json'}")
    print(f"Saved summary -> {save_dir / 'summary.json'}")
    print(f"Saved figure  -> {save_dir / 'functional_similarity_interference_summary.png'}")
    for condition in sorted(summary.keys(), key=condition_sort_key):
        tests = summary[condition]["tests"]
        print(
            f"  {condition}: "
            f"mean ratio={tests['combined_mean_interference']['similar_over_dissimilar_ratio']:.2f}x | "
            f"total ratio={tests['combined_total_interference']['similar_over_dissimilar_ratio']:.2f}x | "
            f"contention ratio={tests['combined_contention_jaccard']['similar_over_dissimilar_ratio']:.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Similar/dissimilar pairwise rewriting assay with functional importance."
    )

    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.4, 0.7])
    parser.add_argument("--include_random", action="store_true", default=True)
    parser.add_argument("--no_random", dest="include_random", action="store_false")
    parser.add_argument("--target_mean", type=float, default=0.05)
    parser.add_argument("--input_mode", type=str, default="spatial", choices=["spatial"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exc_neurons", type=int, default=4096)
    parser.add_argument("--inh_neurons", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--similar_pairs", type=parse_pair, nargs="+", default=SIMILAR_PAIRS)
    parser.add_argument("--dissimilar_pairs", type=parse_pair, nargs="+", default=DISSIMILAR_PAIRS)

    parser.add_argument("--per_class", type=int, default=10)
    parser.add_argument("--T_on", type=int, default=1000)
    parser.add_argument("--T_off", type=int, default=2000)
    parser.add_argument("--T_skip", type=int, default=100)
    parser.add_argument("--burn_in_on", type=int, default=2000)
    parser.add_argument("--burn_in_off", type=int, default=3000)

    parser.add_argument("--top_k_ee", type=int, default=4000)
    parser.add_argument("--top_k_input", type=int, default=2000)
    parser.add_argument(
        "--include_delta_top5_reference",
        action="store_true",
        help="Also store reference metrics for the top-5%%-by-|Delta W_A| important-synapse definition.",
    )

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="outputs/pairwise_rewriting")

    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
