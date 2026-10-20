"""Figure 4 (assembly overlap versus synaptic contention) and Figure A6 (contention versus
directional interference).

The 35 runs are five per connectivity condition (sigma=0.4: seeds 42-46; all other
conditions: seeds 47, 48, 49, 58, 59). Contention and directional interference are the
means of the recurrent (E->E) and input-to-E pathway values.

Input:  results/adjudication/run_records.csv
Output: figures/fig4_adjudication.{png,pdf}, figures/figA6_interference_bridge.{png,pdf}

Usage: python -m paper_figures.fig4_adjudication
"""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from paper_figures.common import RESULTS, reset_style, save, sd

STYLE = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

CONDITIONS = [
    ("distance", 0.4, r"$\sigma=0.4$"),
    ("distance", 0.5, r"$\sigma=0.5$"),
    ("distance", 0.7, r"$\sigma=0.7$"),
    ("distance", 1.0, r"$\sigma=1.0$"),
    ("distance", 1.3, r"$\sigma=1.3$"),
    ("distance", 2.0, r"$\sigma=2.0$"),
    ("random", 1.0, "random"),
]
COLORS = {
    ("distance", 0.4): "#D7301F",
    ("distance", 0.5): "#4393C3",
    ("distance", 0.7): "#2166AC",
    ("distance", 1.0): "#92C5DE",
    ("distance", 1.3): "#D1E5F0",
    ("distance", 2.0): "#F4A582",
    ("random", 1.0): "#D6604D",
}
C_GRID = "#D9D9D9"


def load_records() -> list[dict]:
    with (RESULTS / "adjudication" / "run_records.csv").open() as f:
        rows = list(csv.DictReader(f))
    records = []
    for row in rows:
        records.append({
            "mode": row["mode"],
            "sigma": float(row["sigma"]),
            "seed": int(row["seed"]),
            "accuracy": float(row["final_accuracy"]),
            "overlap": float(row["assembly_overlap"]),
            "contention": 0.5 * (float(row["ee_contention"]) + float(row["input_contention"])),
            "interference": 0.5 * (float(row["ee_interference"]) + float(row["input_interference"])),
        })
    return records


def cond_key(rec: dict) -> tuple[str, float]:
    return rec["mode"], rec["sigma"]


def metric_stats(x, y) -> dict:
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    return {"pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
            "spearman_rho": float(spear[0]), "spearman_p": float(spear[1]), "n": len(x)}


def accuracy_stats(records: list[dict], key: str) -> dict:
    x = np.array([rec[key] for rec in records], dtype=float)
    y = np.array([rec["accuracy"] * 100 for rec in records], dtype=float)
    return metric_stats(x, y)


def add_stats(ax, records: list[dict], key: str, loc: tuple[float, float]) -> None:
    st = accuracy_stats(records, key)
    ax.text(loc[0], loc[1], f"Spearman $\\rho$={st['spearman_rho']:.2f}, p={st['spearman_p']:.2g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.3,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CCCCCC"})


def scatter(ax, records: list[dict], key: str, xlabel: str) -> None:
    for mode, sigma, label in CONDITIONS:
        pts = [rec for rec in records if cond_key(rec) == (mode, sigma)]
        x = np.array([rec[key] for rec in pts], dtype=float)
        y = np.array([rec["accuracy"] * 100 for rec in pts], dtype=float)
        ax.scatter(x, y, s=45, color=COLORS[(mode, sigma)], edgecolor="black", linewidth=0.4, alpha=0.45, label=label)
        ax.errorbar([x.mean()], [y.mean()], xerr=[sd(x)], yerr=[sd(y)], fmt="o", ms=9,
                    color=COLORS[(mode, sigma)], mec="black", mew=1.0, capsize=3, alpha=1.0)
    all_x = np.array([rec[key] for rec in records], dtype=float)
    all_y = np.array([rec["accuracy"] * 100 for rec in records], dtype=float)
    slope, intercept, _, _, _ = stats.linregress(all_x, all_y)
    xs = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.8, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final 10-way CIL accuracy (%)")
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=0.8)


def condition_summary(records: list[dict], key: str) -> list[tuple[float, float]]:
    by_cond: dict[tuple[str, float], list[float]] = defaultdict(list)
    for rec in records:
        by_cond[cond_key(rec)].append(rec[key])
    return [(float(np.mean(by_cond[(m, s)])), sd(by_cond[(m, s)])) for m, s, _ in CONDITIONS]


def build_fig4(records: list[dict]) -> None:
    reset_style()
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0), gridspec_kw={"hspace": 0.42, "wspace": 0.28})
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    x = np.arange(len(CONDITIONS))
    labels = [label for _, _, label in CONDITIONS]
    colors = [COLORS[(mode, sigma)] for mode, sigma, _ in CONDITIONS]

    for ax, key, ylabel, title in (
        (ax_a, "overlap", "Mean assembly overlap", "(a) Assembly overlap varies weakly"),
        (ax_b, "contention", "Mean synaptic contention", "(b) Synaptic contention separates regimes"),
    ):
        summary = condition_summary(records, key)
        ax.bar(x, [m for m, _ in summary], 0.42, yerr=[s for _, s in summary], color=colors,
               edgecolor="black", linewidth=0.8, capsize=4, alpha=0.85, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", color=C_GRID, linewidth=0.8, alpha=0.8)

    scatter(ax_c, records, "overlap", "Mean assembly overlap")
    ax_c.set_title("(c) Assembly overlap weakly predicts accuracy")
    add_stats(ax_c, records, "overlap", (0.97, 0.97))
    ax_c.legend(fontsize=7, loc="lower left", framealpha=0.9)

    scatter(ax_d, records, "contention", "Mean synaptic contention")
    ax_d.set_title("(d) Synaptic contention predicts accuracy")
    add_stats(ax_d, records, "contention", (0.97, 0.97))

    save(fig, "fig4_adjudication", dpi=220)
    plt.close(fig)


def build_figA6(records: list[dict]) -> dict:
    reset_style()
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sigma_rank = {0.4: 0, 0.5: 1, 0.7: 2, 1.0: 3, 1.3: 4, 2.0: 5}
    ordered = sorted(records, key=lambda r: (1, 99, r["seed"]) if r["mode"] == "random"
                     else (0, sigma_rank[r["sigma"]], r["seed"]))
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in ordered:
        label = "random" if rec["mode"] == "random" else f"σ={rec['sigma']}"
        color = COLORS[cond_key(rec)] if rec["mode"] == "distance" else "#D6604D"
        marker = "s" if rec["mode"] == "random" else "o"
        ax.scatter(rec["contention"], rec["interference"], s=54, c=color, marker=marker, edgecolor="black",
                   linewidth=0.5, alpha=0.45, zorder=3, label=None if groups[label] else label)
        groups[label].append(rec)
    for label, recs in groups.items():
        xv = [r["contention"] for r in recs]
        yv = [r["interference"] for r in recs]
        color = COLORS[cond_key(recs[0])] if recs[0]["mode"] == "distance" else "#D6604D"
        marker = "s" if recs[0]["mode"] == "random" else "o"
        ax.errorbar(np.mean(xv), np.mean(yv), xerr=sd(xv), yerr=sd(yv), fmt=marker, ms=12, color=color,
                    ecolor=color, elinewidth=1.5, capsize=4, markeredgecolor="black", markeredgewidth=1.0,
                    alpha=1.0, zorder=5)
    x = np.array([r["contention"] for r in records])
    y = np.array([r["interference"] for r in records])
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.8, zorder=2)
    rho, rho_p = stats.spearmanr(x, y)
    ax.text(0.97, 0.03, f"Pearson r={r_value:.2f}, p={p_value:.3g}\nSpearman ρ={rho:.2f}, p={rho_p:.3g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#BBBBBB"))
    ax.set_xlabel("Mean synaptic contention (Jaccard)")
    ax.set_ylabel("Mean directional interference")
    ax.set_title("Contention tracks directional interference")
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=0.8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "figA6_interference_bridge", dpi=200)
    plt.close(fig)
    return metric_stats(x, y)


def main() -> None:
    records = load_records()
    build_fig4(records)
    bridge = build_figA6(records)
    print(f"n = {len(records)} runs")
    for key in ("overlap", "contention"):
        st = accuracy_stats(records, key)
        print(f"{key:>10} vs accuracy: Pearson r={st['pearson_r']:.2f} (p={st['pearson_p']:.3g}), "
              f"Spearman rho={st['spearman_rho']:.2f} (p={st['spearman_p']:.2g})")
        for (mode, sigma, _), (m, s) in zip(CONDITIONS, condition_summary(records, key)):
            print(f"    {mode} {sigma}: {m:.3f} ± {s:.3f}")
    inter = accuracy_stats(records, "interference")
    print(f"interference vs accuracy: Spearman rho={inter['spearman_rho']:.2f}")
    print(f"contention vs interference: Pearson r={bridge['pearson_r']:.2f}, Spearman rho={bridge['spearman_rho']:.2f}")


if __name__ == "__main__":
    main()
