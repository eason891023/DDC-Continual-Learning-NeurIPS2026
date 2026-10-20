"""
Benchmark-level directional interference analysis from per-task snapshots.

This script consumes the task snapshots saved by:

    python -m experiments.cl_sigma_sweep ... --save_task_snapshots

and measures how much learning a later task perturbs the synapses that were
important for an earlier task. Unlike the two-class pairwise rewriting assay,
this operates directly on the benchmark runs.

Default design:
  - consecutive task pairs only (task t -> task t+1)
  - importance is class-specific and topology-aware:
      EE:     |w_ij| * a_i(c) * a_j(c)
      input:  |w_pj| * a_j(c)
    where a_k(c) is the normalized class affinity of neuron k from
    `neurons_labels[:, c]`
  - directional interference = mean(|Δw| on top-k important synapses)
  - contention = Jaccard overlap between task-A and task-B important synapses

Outputs:
  - pairwise records JSON
  - run-level summary JSON
  - aggregate JSON
  - plots:
      benchmark_interference_bar.(png|pdf)
      benchmark_interference_vs_accuracy.(png|pdf)
      benchmark_contention_vs_interference.(png|pdf)

Usage:
    python -m experiments.benchmark_interference \
      --runs_dir outputs/snapshot_runs/runs \
      --include_random --sigmas 0.5 0.7 1.0 1.3 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    parser = argparse.ArgumentParser(description="Benchmark-level directional interference from task snapshots")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="outputs/snapshot_runs/runs",
        help="Directory containing per-run result.json and task_snapshots/.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/benchmark_interference",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--readout",
        type=str,
        default="selective",
        choices=["selective"],
        help="Which final accuracy field from result.json to align against.",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="*",
        default=None,
        help="Optional whitelist of distance sigmas to include.",
    )
    parser.add_argument(
        "--include_random",
        action="store_true",
        help="Include random runs.",
    )
    parser.add_argument(
        "--synapse_types",
        type=str,
        nargs="+",
        default=["ee", "input"],
        choices=["ee", "input"],
        help="Synapse families to analyze.",
    )
    parser.add_argument(
        "--task_lag",
        type=int,
        default=1,
        help="Compare task t to task t + lag. Default 1 = consecutive tasks only.",
    )
    parser.add_argument(
        "--top_k_ee",
        type=int,
        default=4000,
        help="Top-k important EE synapses per task.",
    )
    parser.add_argument(
        "--top_k_input",
        type=int,
        default=2000,
        help="Top-k important input->EE synapses per task.",
    )
    parser.add_argument(
        "--min_pairs_per_run",
        type=int,
        default=3,
        help="Minimum valid task-pairs required to keep a run in aggregate summaries.",
    )
    return parser.parse_args()


def canonical_sigma_str(sigma: float) -> str:
    text = f"{sigma:.6g}"
    if "." not in text:
        text += ".0"
    return text


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def class_affinity(neurons_labels: torch.Tensor, cls: int) -> torch.Tensor:
    """Normalize class support across neurons."""
    v = neurons_labels[:, cls].float().clamp(min=0)
    denom = v.sum().clamp(min=1e-12)
    return v / denom


def task_snapshot_dirs(run_dir: Path) -> List[Path]:
    snap_root = run_dir / "task_snapshots"
    if not snap_root.exists():
        return []
    return sorted([p for p in snap_root.iterdir() if p.is_dir()])


def resolve_snapshot_run_dir(run_dir: Path, mode: str, sigma: float, seed: int) -> Path:
    """
    Result JSONs and task snapshots may live in differently named sibling
    directories because older sweep code saved `result.json` under `dist_σ=*`
    while the training artifacts were written under `distance_sigma*`.
    """
    direct = run_dir / "task_snapshots"
    if direct.exists():
        return run_dir

    runs_dir = run_dir.parent

    # Extract order suffix from run_dir name if present (e.g. "_order0123456789")
    import re
    order_match = re.search(r"_(order\d+)", run_dir.name)
    order_suffix = f"_{order_match.group(1)}" if order_match else ""

    if mode == "random":
        candidates = [
            runs_dir / f"random_seed{seed}",
            runs_dir / f"random{order_suffix}_seed{seed}",
        ]
    else:
        sigma_str = canonical_sigma_str(sigma)
        candidates = [
            runs_dir / f"distance_sigma{sigma_str}_seed{seed}",
            runs_dir / f"distance_sigma{sigma}_seed{seed}",
            runs_dir / f"distance_sigma{sigma_str}{order_suffix}_seed{seed}",
            runs_dir / f"distance_sigma{sigma}{order_suffix}_seed{seed}",
        ]

    for candidate in candidates:
        if (candidate / "task_snapshots").exists():
            return candidate

    return run_dir


def load_task_snapshot(snapshot_dir: Path) -> dict:
    meta = load_json(snapshot_dir / "meta.json")
    payload = {
        "meta": meta,
        "neurons_labels": torch.load(snapshot_dir / "neurons_labels.pt", map_location="cpu").float(),
    }

    ee_path = snapshot_dir / "RSNN_EE_weights.pt"
    if ee_path.exists():
        payload["ee_weights"] = torch.load(ee_path, map_location="cpu").float()

    input_path = snapshot_dir / "inputs2RSNN_EE_weights.pt"
    if input_path.exists():
        payload["input_weights"] = torch.load(input_path, map_location="cpu").float()

    return payload


def topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(values.numel()))
    if k <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.topk(values, k=k).indices


def compute_ee_importance_indices(weights: torch.Tensor, neurons_labels: torch.Tensor, cls: int, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = weights.abs() > 1e-12
    pre_idx, post_idx = mask.nonzero(as_tuple=True)
    if pre_idx.numel() == 0:
        return pre_idx, torch.empty(0, dtype=torch.long)

    aff = class_affinity(neurons_labels, cls)
    scores = weights[pre_idx, post_idx].abs() * aff[pre_idx] * aff[post_idx]
    idx = topk_indices(scores, top_k)
    flat = pre_idx[idx] * weights.shape[1] + post_idx[idx]
    return flat.cpu(), idx.cpu()


def compute_input_importance_indices(weights: torch.Tensor, neurons_labels: torch.Tensor, cls: int, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = weights.abs() > 1e-12
    inp_idx, post_idx = mask.nonzero(as_tuple=True)
    if inp_idx.numel() == 0:
        return inp_idx, torch.empty(0, dtype=torch.long)

    aff = class_affinity(neurons_labels, cls)
    scores = weights[inp_idx, post_idx].abs() * aff[post_idx]
    idx = topk_indices(scores, top_k)
    flat = inp_idx[idx] * weights.shape[1] + post_idx[idx]
    return flat.cpu(), idx.cpu()


def flat_delta_lookup(weights_a: torch.Tensor, weights_b: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
    width = weights_a.shape[1]
    row = torch.div(flat_idx, width, rounding_mode="floor")
    col = flat_idx % width
    return (weights_b[row, col] - weights_a[row, col]).abs()


def jaccard_from_flat_indices(a: torch.Tensor, b: torch.Tensor) -> float:
    set_a = set(a.tolist())
    set_b = set(b.tolist())
    union = set_a | set_b
    if not union:
        return float("nan")
    return len(set_a & set_b) / len(union)


def mean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def parse_mode_sigma_seed(run_dir: Path, result: dict) -> Tuple[str, float, int]:
    return result["mode"], float(result["sigma"]), int(result["seed"])


def keep_run(mode: str, sigma: float, args: argparse.Namespace) -> bool:
    if mode == "random":
        return args.include_random
    if args.sigmas is None:
        return True
    return any(abs(sigma - allowed) < 1e-9 for allowed in args.sigmas)


def analyze_run(run_dir: Path, args: argparse.Namespace) -> Tuple[List[dict], Optional[dict]]:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return [], None

    result = load_json(result_path)
    mode, sigma, seed = parse_mode_sigma_seed(run_dir, result)
    if not keep_run(mode, sigma, args):
        return [], None

    snapshot_run_dir = resolve_snapshot_run_dir(run_dir, mode, sigma, seed)
    snapshots = task_snapshot_dirs(snapshot_run_dir)
    if not snapshots:
        return [], None

    accuracy_key = f"final_acc_{args.readout}"
    final_acc = float(result[accuracy_key])

    payloads = [load_task_snapshot(p) for p in snapshots]
    pair_records: List[dict] = []

    for t in range(len(payloads) - args.task_lag):
        a = payloads[t]
        b = payloads[t + args.task_lag]
        cls_a = int(a["meta"]["class_learned"])
        cls_b = int(b["meta"]["class_learned"])

        rec = {
            "run_dir": str(run_dir),
            "snapshot_run_dir": str(snapshot_run_dir),
            "mode": mode,
            "sigma": sigma,
            "seed": seed,
            "task_a": int(a["meta"]["task_idx"]),
            "task_b": int(b["meta"]["task_idx"]),
            "class_a": cls_a,
            "class_b": cls_b,
            "lag": args.task_lag,
            "final_accuracy": final_acc,
        }

        if "ee" in args.synapse_types and "ee_weights" in a and "ee_weights" in b:
            flat_a, _ = compute_ee_importance_indices(a["ee_weights"], a["neurons_labels"], cls_a, args.top_k_ee)
            flat_b, _ = compute_ee_importance_indices(b["ee_weights"], b["neurons_labels"], cls_b, args.top_k_ee)
            if flat_a.numel() > 0 and flat_b.numel() > 0:
                delta_a = flat_delta_lookup(a["ee_weights"], b["ee_weights"], flat_a)
                rec["ee_directional_interference"] = float(delta_a.mean().item())
                rec["ee_directional_interference_median"] = float(delta_a.median().item())
                rec["ee_contention_jaccard"] = float(jaccard_from_flat_indices(flat_a, flat_b))
                rec["ee_topk"] = int(flat_a.numel())

        if "input" in args.synapse_types and "input_weights" in a and "input_weights" in b:
            flat_a, _ = compute_input_importance_indices(a["input_weights"], a["neurons_labels"], cls_a, args.top_k_input)
            flat_b, _ = compute_input_importance_indices(b["input_weights"], b["neurons_labels"], cls_b, args.top_k_input)
            if flat_a.numel() > 0 and flat_b.numel() > 0:
                delta_a = flat_delta_lookup(a["input_weights"], b["input_weights"], flat_a)
                rec["input_directional_interference"] = float(delta_a.mean().item())
                rec["input_directional_interference_median"] = float(delta_a.median().item())
                rec["input_contention_jaccard"] = float(jaccard_from_flat_indices(flat_a, flat_b))
                rec["input_topk"] = int(flat_a.numel())

        pair_records.append(rec)

    if len(pair_records) < args.min_pairs_per_run:
        return pair_records, None

    run_summary = {
        "run_dir": str(run_dir),
        "snapshot_run_dir": str(snapshot_run_dir),
        "mode": mode,
        "sigma": sigma,
        "seed": seed,
        "final_accuracy": final_acc,
        "n_pairs": len(pair_records),
    }
    for prefix in ("ee", "input"):
        di_key = f"{prefix}_directional_interference"
        cj_key = f"{prefix}_contention_jaccard"
        di_vals = [r[di_key] for r in pair_records if di_key in r]
        cj_vals = [r[cj_key] for r in pair_records if cj_key in r]
        if di_vals:
            run_summary[f"{prefix}_directional_interference_mean"] = mean_or_nan(di_vals)
        if cj_vals:
            run_summary[f"{prefix}_contention_jaccard_mean"] = mean_or_nan(cj_vals)

    if (
        "ee_directional_interference_mean" in run_summary and
        "input_directional_interference_mean" in run_summary
    ):
        run_summary["combined_interference_mean"] = 0.5 * (
            run_summary["ee_directional_interference_mean"] +
            run_summary["input_directional_interference_mean"]
        )

    return pair_records, run_summary


def aggregate_runs(run_summaries: List[dict]) -> dict:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for rec in run_summaries:
        key = "random" if rec["mode"] == "random" else f"sigma={canonical_sigma_str(rec['sigma'])}"
        grouped[key].append(rec)

    agg = {}
    for key, vals in grouped.items():
        entry = {"n_runs": len(vals)}
        for metric in (
            "final_accuracy",
            "ee_directional_interference_mean",
            "input_directional_interference_mean",
            "combined_interference_mean",
            "ee_contention_jaccard_mean",
            "input_contention_jaccard_mean",
        ):
            arr = [v[metric] for v in vals if metric in v]
            if arr:
                entry[metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "values": arr,
                }
        agg[key] = entry
    return agg


def scatter_with_fit(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    colors: List[str],
    markers: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
) -> dict:
    for x, y, c, m in zip(xs, ys, colors, markers):
        ax.scatter(x, y, s=58, c=c, marker=m, edgecolor="black", linewidth=0.5, alpha=0.85)

    if len(xs) >= 2:
        slope, intercept, r_value, p_value, _ = stats.linregress(xs, ys)
        line_x = np.linspace(float(xs.min()), float(xs.max()), 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color="#333333", linewidth=1.8)
        rho, rho_p = stats.spearmanr(xs, ys)
        text = f"Pearson r={r_value:.2f}, p={p_value:.3g}\nSpearman rho={rho:.2f}, p={rho_p:.3g}"
        out = {
            "pearson_r": float(r_value),
            "pearson_p": float(p_value),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
        }
    else:
        text = "Not enough runs"
        out = {}

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
    return out


def plot_bar(agg: dict, save_dir: Path) -> None:
    keys = sorted(agg.keys(), key=lambda k: (k == "random", k))
    metrics = [
        ("ee_directional_interference_mean", "EE directional interference"),
        ("input_directional_interference_mean", "Input directional interference"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, (metric, title) in zip(axes, metrics):
        xs, ys, es, colors = [], [], [], []
        labels = []
        for idx, key in enumerate(keys):
            if metric not in agg[key]:
                continue
            xs.append(idx)
            ys.append(agg[key][metric]["mean"])
            es.append(agg[key][metric]["std"])
            colors.append(C_RAND if key == "random" else C_DIST)
            labels.append(key)

        ax.bar(np.arange(len(ys)), ys, yerr=es, color=colors, edgecolor="black", linewidth=0.6, capsize=4)
        ax.set_xticks(np.arange(len(ys)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8, alpha=0.8)

    fig.suptitle("Benchmark-level directional interference", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "benchmark_interference_bar.png", bbox_inches="tight")
    fig.savefig(save_dir / "benchmark_interference_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_run_scatters(run_summaries: List[dict], save_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    colors = [C_RAND if r["mode"] == "random" else C_DIST for r in run_summaries]
    markers = ["s" if r["mode"] == "random" else "o" for r in run_summaries]
    acc = np.array([r["final_accuracy"] * 100.0 for r in run_summaries], dtype=float)

    stats_out = {}

    if all("combined_interference_mean" in r for r in run_summaries):
        xs = np.array([r["combined_interference_mean"] for r in run_summaries], dtype=float)
        stats_out["interference_vs_accuracy"] = scatter_with_fit(
            axes[0],
            xs=xs,
            ys=acc,
            colors=colors,
            markers=markers,
            xlabel="Mean directional interference",
            ylabel="Final 10-way CIL accuracy (%)",
            title="(a) Less interference, better CL",
        )

    if all(
        ("combined_interference_mean" in r and "ee_contention_jaccard_mean" in r and "input_contention_jaccard_mean" in r)
        for r in run_summaries
    ):
        contention = np.array([
            0.5 * (r["ee_contention_jaccard_mean"] + r["input_contention_jaccard_mean"])
            for r in run_summaries
        ], dtype=float)
        interference = np.array([r["combined_interference_mean"] for r in run_summaries], dtype=float)
        stats_out["contention_vs_interference"] = scatter_with_fit(
            axes[1],
            xs=contention,
            ys=interference,
            colors=colors,
            markers=markers,
            xlabel="Mean contention (Jaccard overlap of important synapses)",
            ylabel="Mean directional interference",
            title="(b) More contention, more interference",
        )

    fig.tight_layout()
    fig.savefig(save_dir / "benchmark_interference_vs_accuracy.png", bbox_inches="tight")
    fig.savefig(save_dir / "benchmark_interference_vs_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

    return stats_out


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pair_records: List[dict] = []
    run_summaries: List[dict] = []

    run_dirs = sorted([p.parent for p in runs_dir.glob("**/result.json")])
    for run_dir in run_dirs:
        pairs, summary = analyze_run(run_dir, args)
        pair_records.extend(pairs)
        if summary is not None:
            run_summaries.append(summary)

    if not run_summaries:
        raise SystemExit(
            "No valid runs found. Make sure your sweep was run with --save_task_snapshots "
            "and that --runs_dir points to the run directories."
        )

    agg = aggregate_runs(run_summaries)
    stats_out = plot_run_scatters(run_summaries, save_dir)
    plot_bar(agg, save_dir)

    (save_dir / "pair_records.json").write_text(json.dumps(pair_records, indent=2))
    (save_dir / "run_summaries.json").write_text(json.dumps(run_summaries, indent=2))
    summary = {
        "config": {
            "runs_dir": args.runs_dir,
            "readout": args.readout,
            "sigmas": args.sigmas,
            "include_random": args.include_random,
            "task_lag": args.task_lag,
            "synapse_types": args.synapse_types,
            "top_k_ee": args.top_k_ee,
            "top_k_input": args.top_k_input,
        },
        "n_pair_records": len(pair_records),
        "n_run_summaries": len(run_summaries),
        "aggregate": agg,
        "stats": stats_out,
    }
    (save_dir / "aggregate.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved pair records to {save_dir / 'pair_records.json'}")
    print(f"Saved run summaries to {save_dir / 'run_summaries.json'}")
    print(f"Saved aggregate summary to {save_dir / 'aggregate.json'}")
    print(f"Analyzed {len(run_summaries)} runs and {len(pair_records)} task-pairs")

    for key, val in agg.items():
        msg = [f"  {key:<10} n={val['n_runs']}"]
        if "final_accuracy" in val:
            msg.append(
                f"acc={val['final_accuracy']['mean']*100:.2f}±{val['final_accuracy']['std']*100:.2f}"
            )
        if "combined_interference_mean" in val:
            msg.append(
                f"comb_interf={val['combined_interference_mean']['mean']:.6f}±{val['combined_interference_mean']['std']:.6f}"
            )
        if "ee_contention_jaccard_mean" in val:
            msg.append(
                f"ee_cont={val['ee_contention_jaccard_mean']['mean']:.4f}"
            )
        print(" ".join(msg))


if __name__ == "__main__":
    main()
