"""
Money plot: connect mechanism proxies to continual-learning performance.

This script reads run-level class-incremental results plus the saved
`neurons_labels.pt` tensors saved by `experiments.cl_sigma_sweep`.
It then builds simple mechanism proxies directly from the benchmark runs:

  1. Readout assembly overlap proxy:
     For each class, take the top-k neurons in `neurons_labels[:, class]`
     and compute the mean pairwise Jaccard overlap across classes.

  2. Neuron selectivity proxy:
     Treat each neuron's class-wise spike totals as a response profile and
     compute an entropy-based selectivity index.

  3. Sharing proxy:
     Count how many classes each active neuron participates in on the
     accumulated readout tensor.

  4. Headroom proxy:
     Separation × sharing. This captures the intuition that low baseline
     overlap provides room for beneficial sharing without destructive
     interference.

These are not identical to the toy Step-2 metrics, but they are useful for
showing a cleaner mechanism-to-performance link on the actual benchmark runs.
In practice this gives us both sides of the story:

  - separation proxy: low overlap
  - sharing proxy: multi-class participation

Usage:
    python -m experiments.money_plot
    python -m experiments.money_plot --runs_dir outputs/snapshot_runs/runs
    python -m experiments.money_plot --top_k 100 --readout selective
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


C_DIST = "#2166AC"
C_RAND = "#D6604D"
C_GRID = "#D9D9D9"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mechanism-to-performance money plot")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="outputs/snapshot_runs/runs",
        help="Directory containing per-run result.json and neurons_labels.pt files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/assembly_overlap",
        help="Directory for output figures and summary JSON.",
    )
    parser.add_argument(
        "--readout",
        type=str,
        default="selective",
        choices=["selective"],
        help="Which final accuracy field to use from result.json.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-k neurons per class used to form the overlap proxy.",
    )
    parser.add_argument(
        "--min_active_per_class",
        type=int,
        default=10,
        help="Skip runs whose per-class supports are too degenerate.",
    )
    parser.add_argument(
        "--include_random",
        action="store_true",
        help="Include random runs in the scatter plots. Off by default for a cleaner sigma story.",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="*",
        default=None,
        help="Optional whitelist of distance sigmas to include, e.g. --sigmas 0.5 0.7 1.0 1.3",
    )
    return parser.parse_args()


def canonical_sigma_str(sigma: float) -> str:
    text = f"{sigma:.6g}"
    if "." not in text:
        text += ".0"
    return text


def find_neurons_labels_path(runs_dir: Path, result_dir: Path, mode: str, sigma: float, seed: int) -> Optional[Path]:
    direct = result_dir / "neurons_labels.pt"
    if direct.exists():
        return direct

    # Extract order suffix from result_dir name if present (e.g. "_order0123456789")
    import re
    order_match = re.search(r"_(order\d+)", result_dir.name)
    order_suffix = f"_{order_match.group(1)}" if order_match else ""

    if mode == "random":
        candidates = [
            runs_dir / f"random_seed{seed}" / "neurons_labels.pt",
            runs_dir / f"random{order_suffix}_seed{seed}" / "neurons_labels.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    sigma_str = canonical_sigma_str(sigma)
    candidates = [
        runs_dir / f"distance_sigma{sigma_str}_seed{seed}" / "neurons_labels.pt",
        runs_dir / f"distance_sigma{sigma}_seed{seed}" / "neurons_labels.pt",
        runs_dir / f"distance_sigma{sigma_str}{order_suffix}_seed{seed}" / "neurons_labels.pt",
        runs_dir / f"distance_sigma{sigma}{order_suffix}_seed{seed}" / "neurons_labels.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def compute_mean_overlap_topk(neurons_labels: torch.Tensor, top_k: int) -> float:
    n_classes = neurons_labels.shape[1]
    assemblies: List[set] = []

    for cls in range(n_classes):
        scores = neurons_labels[:, cls]
        active = int((scores > 0).sum().item())
        k = min(top_k, active if active > 0 else top_k)
        if k <= 0:
            assemblies.append(set())
            continue
        idx = torch.topk(scores, k=k).indices.cpu().tolist()
        assemblies.append(set(idx))

    overlaps: List[float] = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            union = assemblies[i] | assemblies[j]
            if not union:
                continue
            overlaps.append(len(assemblies[i] & assemblies[j]) / len(union))

    return float(np.mean(overlaps)) if overlaps else float("nan")


def compute_mean_selectivity(neurons_labels: torch.Tensor) -> float:
    response = neurons_labels.float()
    total = response.sum(dim=1, keepdim=True)
    active_mask = total.squeeze(1) > 0
    if int(active_mask.sum()) == 0:
        return float("nan")

    p = response[active_mask] / total[active_mask].clamp(min=1e-12)
    entropy = -(p * torch.log(p.clamp(min=1e-12))).sum(dim=1)
    max_entropy = math.log(response.shape[1])
    selectivity = 1.0 - entropy / max_entropy
    return float(selectivity.mean().item())


def compute_classes_per_neuron(neurons_labels: torch.Tensor) -> float:
    active = neurons_labels.sum(dim=1) > 0
    if int(active.sum()) == 0:
        return float("nan")
    counts = (neurons_labels[active] > 0).sum(dim=1).float()
    return float(counts.mean().item())


def load_run_records(args: argparse.Namespace) -> List[Dict[str, float]]:
    runs_dir = Path(args.runs_dir)
    result_paths = sorted(runs_dir.glob("**/result.json"))
    accuracy_key = f"final_acc_{args.readout}"

    records: List[Dict[str, float]] = []
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        mode = result["mode"]
        if mode == "random" and not args.include_random:
            continue

        sigma = float(result["sigma"])
        if mode == "distance" and args.sigmas is not None:
            if not any(abs(sigma - allowed) < 1e-9 for allowed in args.sigmas):
                continue
        seed = int(result["seed"])
        if accuracy_key not in result:
            continue

        labels_path = find_neurons_labels_path(
            runs_dir=runs_dir,
            result_dir=result_path.parent,
            mode=mode,
            sigma=sigma,
            seed=seed,
        )
        if labels_path is None:
            print(f"[skip] no neurons_labels.pt found for {result_path}")
            continue

        neurons_labels = torch.load(labels_path, map_location="cpu")
        if not isinstance(neurons_labels, torch.Tensor) or neurons_labels.ndim != 2:
            print(f"[skip] unexpected neurons_labels format: {labels_path}")
            continue

        min_active = int((neurons_labels > 0).sum(dim=0).min().item())
        if min_active < args.min_active_per_class:
            print(f"[skip] degenerate run (min active/class={min_active}): {result_path}")
            continue

        mean_overlap = compute_mean_overlap_topk(neurons_labels, top_k=args.top_k)
        mean_selectivity = compute_mean_selectivity(neurons_labels)
        mean_classes_per_neuron = compute_classes_per_neuron(neurons_labels)

        record = {
            "mode": mode,
            "sigma": sigma,
            "seed": seed,
            "accuracy": float(result[accuracy_key]),
            "mean_overlap_topk": mean_overlap,
            "separation_score": 1.0 - mean_overlap if not math.isnan(mean_overlap) else float("nan"),
            "mean_selectivity": mean_selectivity,
            "mean_classes_per_neuron": mean_classes_per_neuron,
            "headroom_score": (
                (1.0 - mean_overlap) * mean_classes_per_neuron
                if not math.isnan(mean_overlap) and not math.isnan(mean_classes_per_neuron)
                else float("nan")
            ),
            "n_active_neurons": int((neurons_labels.sum(dim=1) > 0).sum().item()),
            "labels_path": str(labels_path),
            "result_path": str(result_path),
        }
        records.append(record)

    return records


def fit_line(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    if len(x) < 2:
        return None
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ys = slope * xs + intercept
    return {
        "xs": xs,
        "ys": ys,
        "slope": slope,
        "intercept": intercept,
        "r": r_value,
        "p": p_value,
    }


def summarize_groups(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, float]]] = {}
    for rec in records:
        key = "random" if rec["mode"] == "random" else f"sigma={canonical_sigma_str(rec['sigma'])}"
        groups.setdefault(key, []).append(rec)

    summary: Dict[str, Dict[str, float]] = {}
    for key, vals in groups.items():
        summary[key] = {
            "n": len(vals),
            "accuracy_mean": float(np.mean([v["accuracy"] for v in vals])),
            "accuracy_std": float(np.std([v["accuracy"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
            "overlap_mean": float(np.mean([v["mean_overlap_topk"] for v in vals])),
            "overlap_std": float(np.std([v["mean_overlap_topk"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
            "selectivity_mean": float(np.mean([v["mean_selectivity"] for v in vals])),
            "selectivity_std": float(np.std([v["mean_selectivity"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
            "sharing_mean": float(np.mean([v["mean_classes_per_neuron"] for v in vals])),
            "sharing_std": float(np.std([v["mean_classes_per_neuron"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
            "headroom_mean": float(np.mean([v["headroom_score"] for v in vals])),
            "headroom_std": float(np.std([v["headroom_score"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
        }
    return summary


def plot_metric_vs_accuracy(
    ax: plt.Axes,
    records: List[Dict[str, float]],
    x_key: str,
    x_label: str,
    panel_title: str,
) -> Dict[str, float]:
    x = np.array([rec[x_key] for rec in records], dtype=float)
    y = np.array([rec["accuracy"] * 100.0 for rec in records], dtype=float)

    for rec in records:
        color = C_RAND if rec["mode"] == "random" else C_DIST
        marker = "s" if rec["mode"] == "random" else "o"
        alpha = 0.65 if rec["mode"] == "random" else 0.85
        ax.scatter(
            rec[x_key],
            rec["accuracy"] * 100.0,
            s=54,
            c=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.5,
            alpha=alpha,
            zorder=3,
        )

    line = fit_line(x, y)
    if line is not None:
        ax.plot(line["xs"], line["ys"], color="#333333", linewidth=1.8, zorder=2)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        text = f"Pearson r={line['r']:.2f}, p={line['p']:.3g}\nSpearman ρ={spearman_rho:.2f}, p={spearman_p:.3g}"
    else:
        spearman_rho, spearman_p = float("nan"), float("nan")
        text = "Not enough points for fit"

    ax.set_xlabel(x_label)
    ax.set_ylabel("Final 10-way CIL accuracy (%)")
    ax.set_title(panel_title)
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=0.8)
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
    )
    return {
        "pearson_r": float(line["r"]) if line is not None else float("nan"),
        "pearson_p": float(line["p"]) if line is not None else float("nan"),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def make_figure(args: argparse.Namespace, records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    overlap_stats = plot_metric_vs_accuracy(
        axes[0],
        records,
        x_key="mean_overlap_topk",
        x_label=f"Mean class-overlap proxy (top-{args.top_k} Jaccard)",
        panel_title="(a) Less overlap, better CL",
    )

    sharing_stats = plot_metric_vs_accuracy(
        axes[1],
        records,
        x_key="mean_classes_per_neuron",
        x_label="Mean classes per active neuron",
        panel_title="(b) Controlled sharing supports CL",
    )

    headroom_stats = plot_metric_vs_accuracy(
        axes[2],
        records,
        x_key="headroom_score",
        x_label="Headroom score = (1 - overlap) x sharing",
        panel_title="(c) Separation headroom best tracks CL",
    )

    title = "Money Plot: mechanism proxies tracked on benchmark runs"
    if not args.include_random:
        title += " (distance conditions only)"
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()

    stem = save_dir / "money_plot"
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "overlap_vs_accuracy": overlap_stats,
        "sharing_vs_accuracy": sharing_stats,
        "headroom_vs_accuracy": headroom_stats,
        "group_summary": summarize_groups(records),
    }


def main() -> None:
    args = parse_args()
    records = load_run_records(args)
    if not records:
        raise SystemExit("No usable runs found. Check --runs_dir and the saved artifacts.")

    stats_summary = make_figure(args, records)

    out = {
        "config": {
            "runs_dir": args.runs_dir,
            "save_dir": args.save_dir,
            "readout": args.readout,
            "top_k": args.top_k,
            "include_random": args.include_random,
            "sigmas": args.sigmas,
        },
        "n_records": len(records),
        "records": records,
        "stats": stats_summary,
    }

    summary_path = Path(args.save_dir) / "money_plot_summary.json"
    summary_path.write_text(json.dumps(out, indent=2))

    print(f"Saved figure: {Path(args.save_dir) / 'money_plot.png'}")
    print(f"Saved summary: {summary_path}")
    print(f"Loaded {len(records)} runs")
    for key, val in stats_summary["group_summary"].items():
        print(
            f"  {key:<10} n={val['n']:<2d} "
            f"acc={val['accuracy_mean']*100:.2f}±{val['accuracy_std']*100:.2f} "
            f"overlap={val['overlap_mean']:.3f}±{val['overlap_std']:.3f} "
            f"selectivity={val['selectivity_mean']:.3f}±{val['selectivity_std']:.3f} "
            f"sharing={val['sharing_mean']:.3f}±{val['sharing_std']:.3f} "
            f"headroom={val['headroom_mean']:.3f}±{val['headroom_std']:.3f}"
        )


if __name__ == "__main__":
    main()
