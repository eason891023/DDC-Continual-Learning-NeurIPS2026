"""Figure 6, Table 1, and the supplementary drift/readout figure.

- Readout ladder (n=5 paired seeds): fixed selective-vote readout, the same readout
  recalibrated on final responses, and a linear decoder refit on final responses
  (old classes, after the full stream).
- Fixed-probe audit (n=3 paired seeds): acquisition accuracy, final old-class accuracy,
  BWT, forgetting, and drift (1 - cosine similarity) of class-mean population responses
  relative to acquisition (selective-vote readout; plasticity off; state reset before every image).
All spreads are sample SDs (ddof=1).

Inputs:  results/readout_ladder/readout_ladder.csv, results/fixed_probe_audit/runs/*.json
Outputs: figures/fig6_drift_readout.{png,pdf}, figures/figA_drift_readout_supplement.{png,pdf},
         figures/tab_acquisition_retention.tex

Usage: python -m paper_figures.fig6_drift_readout
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from paper_figures.common import FIG_DIR, RESULTS, ROOT, reset_style

LADDER_CSV = RESULTS / "readout_ladder" / "readout_ladder.csv"
FIXED_PROBE_RUNS = RESULTS / "fixed_probe_audit" / "runs"
APPENDIX_FIG_DIR = FIG_DIR
TABLE_DIR = FIG_DIR

CONDITIONS = ["distance_sigma0.4", "distance_sigma0.7", "random"]
LABELS = {"distance_sigma0.4": r"$\sigma=0.4$", "distance_sigma0.7": r"$\sigma=0.7$", "random": "Random"}
# Same condition colors as Fig. 3.
COLORS = {"distance_sigma0.4": "#D7301F", "distance_sigma0.7": "#2166AC", "random": "#A65628"}
TEXT = "#333333"
CONNECTOR = "#8C8C8C"
OLD_CLASSES = range(9)  # classes 0-8; the last class has no later learning
N_TASKS = 10
DRIFT_YLABEL = "Representational drift\n(1 − cosine similarity\nto acquisition response)"
LAG_XLABEL = "Number of new-class acquisitions, N"


def sd(values) -> float:
    return float(np.std(values, ddof=1))


def set_style() -> None:
    reset_style()
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})


def save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- data

def load_ladder() -> list[dict]:
    with LADDER_CSV.open() as f:
        return list(csv.DictReader(f))


def load_fixed_probe_runs() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXED_PROBE_RUNS.glob("*.json"))]


def drift_by_lag(runs: list[dict], classes) -> dict[str, dict[int, list[float]]]:
    """Per seed: mean drift over `classes` at each number of classes learned after acquisition."""
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        per_lag: dict[int, list[float]] = defaultdict(list)
        for rec in run["drift_records"]:
            if rec["class"] in classes:
                per_lag[rec["tasks_since_learning"]].append(rec["cosine_drift"])
        for lag, values in per_lag.items():
            out[run["condition"]][lag].append(float(np.mean(values)))
    return out


def drift_by_stage(runs: list[dict]) -> dict[str, dict[int, list[float]]]:
    """Per seed: after learning class t, mean drift of all earlier classes k < t."""
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        drift = {(r["class"], r["task_idx"]): r["cosine_drift"] for r in run["drift_records"]}
        for t in range(1, N_TASKS):
            out[run["condition"]][t].append(float(np.mean([drift[(k, t)] for k in range(t)])))
    return out


def ladder_by_condition(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        c = row["condition"]
        out[c]["fixed"].append(float(row["frozen_vote_old_accuracy"]) * 100)
        out[c]["recalibrated"].append(float(row["final_vote_old_accuracy"]) * 100)
        out[c]["refit"].append(float(row["final_lr_old_accuracy"]) * 100)
    return out


# ---------------------------------------------------------------- panels

def plot_drift_curves(ax, data: dict[str, dict[int, list[float]]], xs, legend_loc: str | None) -> None:
    for c in CONDITIONS:
        m = np.array([np.mean(data[c][x]) for x in xs])
        e = np.array([sd(data[c][x]) for x in xs])
        ax.fill_between(xs, m - e, m + e, color=COLORS[c], alpha=0.15, linewidth=0)
        ax.plot(xs, m, marker="o", markersize=4, linewidth=1.6, color=COLORS[c], label=LABELS[c])
    ax.set_xticks(list(xs))
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    if legend_loc:
        ax.legend(loc=legend_loc, frameon=False)


def plot_readout_contrast(ax, ladder: dict[str, dict[str, list[float]]]) -> None:
    """Fixed readout vs. refit decoder; each readout's three conditions joined by a dashed line."""
    x = np.arange(len(CONDITIONS))
    for key, hollow in (("fixed", True), ("refit", False)):
        means = np.array([np.mean(ladder[c][key]) for c in CONDITIONS])
        errs = np.array([sd(ladder[c][key]) for c in CONDITIONS])
        ax.plot(x, means, linestyle="--", linewidth=1.1, color=CONNECTOR, zorder=1)
        for xi, c, m, e in zip(x, CONDITIONS, means, errs):
            ax.errorbar(xi, m, yerr=e, fmt="o", markersize=6, mfc="white" if hollow else COLORS[c],
                        mec=COLORS[c], mew=1.5, ecolor=COLORS[c], elinewidth=0.9, capsize=2.5, zorder=2)
            if key == "refit":
                ax.text(xi, m + max(e, 0.8) + 1.6, f"{m:.1f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
            else:
                ax.text(xi, m - e - 1.6, f"{m:.1f}", ha="center", va="top", fontsize=7.5, color=TEXT)
    ax.scatter([], [], s=36, facecolor="white", edgecolor=TEXT, linewidth=1.2, label="Fixed readout (learned at acquisition)")
    ax.scatter([], [], s=36, color=TEXT, label="Linear decoder refit on final responses")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_xlim(-0.45, len(CONDITIONS) - 0.55)
    ax.set_ylim(20, 100)
    ax.set_ylabel("Old-class accuracy (%)")
    ax.grid(axis="x", visible=False)


def plot_paired_gain(ax, rows: list[dict]) -> None:
    gains: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        gains[row["seed"]][row["condition"]] = float(row["linear_decoder_refit_gain"]) * 100
    seeds = sorted(gains)
    x = np.arange(len(CONDITIONS))
    for seed in seeds:
        ys = [gains[seed][c] for c in CONDITIONS]
        ax.plot(x, ys, color="#BDBDBD", linewidth=1.0, zorder=1)
        ax.scatter(x, ys, s=22, color=[COLORS[c] for c in CONDITIONS], edgecolor="white", linewidth=0.6, zorder=2)
    means = [np.mean([gains[s][c] for s in seeds]) for c in CONDITIONS]
    errs = [sd([gains[s][c] for s in seeds]) for c in CONDITIONS]
    ax.errorbar(x + 0.18, means, yerr=errs, fmt="D", color=TEXT, markersize=4.5, capsize=2.5, linewidth=0.9, zorder=3)
    for xi, m in zip(x, means):
        ax.text(xi + 0.27, m, f"{m:.1f}", ha="left", va="center", fontsize=7.5, color=TEXT)
    n_min = sum(min(gains[s], key=gains[s].get) == "distance_sigma0.7" for s in seeds)
    ax.text(0.98, 0.97, rf"$\sigma=0.7$ lowest in {n_min}/{len(seeds)} paired seeds", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_xlim(-0.4, len(CONDITIONS) - 0.25)
    ax.set_ylim(0, max(max(g.values()) for g in gains.values()) + 12)
    ax.set_ylabel("Refit gain (percentage points)")
    ax.grid(axis="x", visible=False)
    print(f"  refit gain: sigma=0.7 lowest in {n_min}/{len(seeds)} seeds;",
          {c: (round(m, 1), round(e, 1)) for c, m, e in zip(CONDITIONS, means, errs)})


# ---------------------------------------------------------------- figures and table

def build_fig6(runs: list[dict], rows: list[dict]) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28})
    lag_curves = drift_by_lag(runs, OLD_CLASSES)
    plot_drift_curves(axes[0], lag_curves, range(N_TASKS), legend_loc="upper left")
    axes[0].set_xlabel(LAG_XLABEL)
    axes[0].set_ylabel(DRIFT_YLABEL)
    axes[0].set_title("(a) Representations drift with later learning")
    ladder = ladder_by_condition(rows)
    plot_readout_contrast(axes[1], ladder)
    axes[1].set_title("(b) Information remains decodable")
    save(fig, FIG_DIR, "fig6_drift_readout")

    print("Fig. 6a drift by lag (mean over 3 seeds), lags 1-9:")
    for c in CONDITIONS:
        print(f"  {c:18s}", " ".join(f"{np.mean(lag_curves[c][l]):.3f}" for l in range(1, N_TASKS)))
    print("Fig. 6b old-class accuracy (mean, SD; n=5):")
    for c in CONDITIONS:
        print(f"  {c:18s}", {k: (round(np.mean(v), 1), round(sd(v), 1)) for k, v in ladder[c].items()})


def build_appendix_figure(runs: list[dict], rows: list[dict]) -> None:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.3), gridspec_kw={"wspace": 0.3})
    stage = drift_by_stage(runs)
    plot_drift_curves(axes[0], stage, range(1, N_TASKS), legend_loc="lower right")
    axes[0].set_xlabel("Class just learned in the stream")
    axes[0].set_ylabel("Mean drift of all earlier classes\n(1 − cosine similarity to acquisition)")
    axes[0].set_title("(a) Earlier classes along the stream")
    digit0 = drift_by_lag(runs, {0})
    plot_drift_curves(axes[1], digit0, range(N_TASKS), legend_loc=None)
    axes[1].set_xlabel(LAG_XLABEL)
    axes[1].set_ylabel(DRIFT_YLABEL)
    axes[1].set_title("(b) First-learned class (digit 0)")
    plot_paired_gain(axes[2], rows)
    axes[2].set_title("(c) Refit gain in each paired seed")
    save(fig, APPENDIX_FIG_DIR, "figA_drift_readout_supplement")

    print("Appendix (a) drift of earlier classes by stage, stages 1-9:")
    for c in CONDITIONS:
        print(f"  {c:18s}", " ".join(f"{np.mean(stage[c][t]):.3f}" for t in range(1, N_TASKS)))
    print("Appendix (b) digit-0 drift by lag, lags 1-9:")
    for c in CONDITIONS:
        print(f"  {c:18s}", " ".join(f"{np.mean(digit0[c][l]):.3f}" for l in range(1, N_TASKS)))


def build_table1(runs: list[dict]) -> None:
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        summary = run["selective_vote"]
        acc = np.array(summary["accuracy_matrix"], dtype=float)  # rows: stage, cols: class
        drift = {(r["class"], r["task_idx"]): r["cosine_drift"] for r in run["drift_records"]}
        last = acc.shape[0] - 1
        s = stats[run["condition"]]
        s["acq"].append(np.mean([acc[k, k] for k in OLD_CLASSES]) * 100)
        s["final"].append(np.mean([acc[last, k] for k in OLD_CLASSES]) * 100)
        s["bwt"].append(summary["mean_bwt_excluding_last"] * 100)
        s["forgetting"].append(summary["mean_forgetting_excluding_last"] * 100)
        s["drift"].append(np.mean([drift[(k, last)] for k in OLD_CLASSES]))

    def cell(values, digits):
        return f"${np.mean(values):.{digits}f} \\pm {sd(values):.{digits}f}$"

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Connectivity & Acquisition (\%) & Final old-class (\%) & BWT (pp) & Forgetting (pp) & Drift \\",
        r"\midrule",
    ]
    for c in CONDITIONS:
        s = stats[c]
        lines.append(f"{LABELS[c]} & {cell(s['acq'], 1)} & {cell(s['final'], 1)} & {cell(s['bwt'], 1)} & "
                     f"{cell(s['forgetting'], 1)} & {cell(s['drift'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "tab_acquisition_retention.tex"
    out.write_text("% Generated by paper_figures/fig6_drift_readout.py -- do not edit by hand.\n" + "\n".join(lines) + "\n")
    print(f"Table 1 written to {out.relative_to(ROOT)}")


def main() -> None:
    runs = load_fixed_probe_runs()
    rows = load_ladder()
    build_fig6(runs, rows)
    build_appendix_figure(runs, rows)
    build_table1(runs)


if __name__ == "__main__":
    main()
