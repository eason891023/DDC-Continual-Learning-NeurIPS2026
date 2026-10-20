"""Figure 3 (MNIST connectivity-range sweep and class-wise profiles) and the Fashion-MNIST
appendix figure.

Inputs:  results/benchmark_mnist/runs.json, results/benchmark_fmnist/runs.json
Outputs: figures/fig3_benchmark.{png,pdf}, figures/figA_fmnist_validation.{png,pdf}

Usage: python -m paper_figures.fig3_benchmark
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from paper_figures.common import RESULTS, read_json, reset_style, save, sd

ORDER = [
    "distance_sigma0.4",
    "distance_sigma0.5",
    "distance_sigma0.7",
    "distance_sigma1.0",
    "distance_sigma1.3",
    "distance_sigma2.0",
    "random",
]
LABELS = {
    "distance_sigma0.4": r"$\sigma=0.4$",
    "distance_sigma0.5": r"$\sigma=0.5$",
    "distance_sigma0.7": r"$\sigma=0.7$",
    "distance_sigma1.0": r"$\sigma=1.0$",
    "distance_sigma1.3": r"$\sigma=1.3$",
    "distance_sigma2.0": r"$\sigma=2.0$",
    "random": "random",
}
COLORS = {
    "distance_sigma0.4": "#D7301F",
    "distance_sigma0.5": "#FC8D59",
    "distance_sigma0.7": "#2166AC",
    "distance_sigma1.0": "#67A9CF",
    "distance_sigma1.3": "#BFD3E6",
    "distance_sigma2.0": "#FDD0A2",
    "random": "#A65628",
}


def set_style() -> None:
    reset_style()
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def group_runs(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    return grouped


def draw_sweep(ax, keys: list[str], means: np.ndarray, stds: np.ndarray, title: str, ylim: float) -> None:
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, color=[COLORS[k] for k in keys], edgecolor="black", linewidth=0.6, capsize=3)
    for xi, mean, err in zip(x, means, stds):
        ax.text(xi, mean + err + 1.6, f"{mean:.2f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, ylim)
    ax.grid(axis="y", alpha=0.25)


def build_fig3() -> dict:
    grouped = group_runs(read_json(RESULTS / "benchmark_mnist" / "runs.json"))
    accs = {k: [r["final_acc_selective"] * 100 for r in grouped[k]] for k in ORDER}
    means = np.array([np.mean(accs[k]) for k in ORDER])
    stds = np.array([sd(accs[k]) for k in ORDER])

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1.08], "wspace": 0.16})
    draw_sweep(axes[0], ORDER, means, stds, "(a) Connectivity-range sweep", ylim=100)

    heatmap_order = ["distance_sigma0.4", "distance_sigma0.7", "random"]
    heatmap_matrix = np.array(
        [
            [np.mean([r["final_per_class_selective"].get(str(c), 0.0) for r in grouped[k]]) * 100 for c in range(10)]
            for k in heatmap_order
        ]
    )
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        xticklabels=[str(c) for c in range(10)],
        yticklabels=[r"$\sigma=0.4$", r"$\sigma=0.7$", "Random"],
        vmin=0,
        vmax=100,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.86, "pad": 0.02},
        ax=axes[1],
    )
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Condition")
    axes[1].set_title("(b) Class-wise failure profiles")
    save(fig, "fig3_benchmark", dpi=240)
    plt.close(fig)
    return {k: {"n": len(accs[k]), "mean": float(m), "sd": float(s)} for k, m, s in zip(ORDER, means, stds)}


def build_fmnist() -> dict:
    grouped = group_runs(read_json(RESULTS / "benchmark_fmnist" / "runs.json"))
    keys = ["distance_sigma0.5", "distance_sigma0.7", "distance_sigma1.0", "random"]
    accs = {k: [r["final_acc_selective"] * 100 for r in grouped[k]] for k in keys}
    means = np.array([np.mean(accs[k]) for k in keys])
    stds = np.array([sd(accs[k]) for k in keys])

    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    draw_sweep(ax, keys, means, stds, "Fashion-MNIST connectivity-range sweep", ylim=70)
    save(fig, "figA_fmnist_validation", dpi=240)
    plt.close(fig)
    return {k: {"n": len(accs[k]), "mean": float(m), "sd": float(s)} for k, m, s in zip(keys, means, stds)}


def main() -> None:
    for name, summary in (("MNIST", build_fig3()), ("Fashion-MNIST", build_fmnist())):
        print(name)
        for k, v in summary.items():
            print(f"  {k:<20} n={v['n']:>2}  {v['mean']:.2f} ± {v['sd']:.2f}%")


if __name__ == "__main__":
    main()
