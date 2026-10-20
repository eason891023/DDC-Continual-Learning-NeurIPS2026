"""Figure 5 (similarity-selective recurrent rewriting) and Figure A7 (per-pair breakdown).

Rewriting is the mean absolute change, while learning class B, of the recurrent E->E
synapses most important for class A (three seeds x three pairs per group).

Inputs:  results/similarity_rewriting/pair_records.csv, results/similarity_rewriting/pixel_similarity.json
Outputs: figures/fig5_similarity_rewriting.{png,pdf}, figures/figA7_pair_breakdown.{png,pdf}

Usage: python -m paper_figures.fig5_similarity
"""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from paper_figures.common import FIG_DIR, RESULTS, read_json, reset_style, save, sd

CONDITIONS = [("sigma0.4", r"$\sigma=0.4$"), ("sigma0.7", r"$\sigma=0.7$"), ("random", "random")]
GROUPS = ["similar", "dissimilar"]
GROUP_COLORS = {"similar": "#7B3294", "dissimilar": "#C2A5CF"}
GROUP_HATCHES = {"similar": "", "dissimilar": "//"}
CONDITION_COLORS = {"sigma0.4": "#4DAF4A", "sigma0.7": "#2166AC", "random": "#D6604D"}


def load_records() -> list[dict]:
    with (RESULTS / "similarity_rewriting" / "pair_records.csv").open() as f:
        rows = list(csv.DictReader(f))
    return [{**r, "class_a": int(r["class_a"]), "class_b": int(r["class_b"]),
             "rewriting": float(r["ee_mean_rewriting"])} for r in rows]


def summarize(records: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for condition, _ in CONDITIONS:
        values = {g: np.array([r["rewriting"] for r in records if r["condition"] == condition and r["group"] == g])
                  for g in GROUPS}
        summary[condition] = {g: {"mean": float(v.mean()), "sd": sd(v), "n": int(v.size)} for g, v in values.items()}
        summary[condition]["ratio"] = float(values["similar"].mean() / values["dissimilar"].mean())
        summary[condition]["mann_whitney_p"] = float(
            stats.mannwhitneyu(values["similar"], values["dissimilar"], alternative="two-sided").pvalue)
    return summary


def build_fig5(summary: dict) -> None:
    reset_style()
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.1, 3.15))
    x = np.arange(len(CONDITIONS), dtype=float)
    width = 0.34
    offsets = {"similar": -width / 2, "dissimilar": width / 2}
    for group in GROUPS:
        means = [summary[c][group]["mean"] for c, _ in CONDITIONS]
        errs = [summary[c][group]["sd"] for c, _ in CONDITIONS]
        ax.bar(x + offsets[group], means, width=width, yerr=errs, capsize=4, color=GROUP_COLORS[group],
               edgecolor="#222222", linewidth=1.0, hatch=GROUP_HATCHES[group], label=group.capitalize(), alpha=0.96)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in CONDITIONS])
    ax.set_ylabel("Mean recurrent rewriting")
    ax.set_title("Similarity-selective recurrent rewriting", fontweight="bold", pad=12)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    top = max(summary[c][g]["mean"] + summary[c][g]["sd"] for c, _ in CONDITIONS for g in GROUPS)
    ax.set_ylim(0, top * 1.05)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig5_similarity_rewriting.{ext}", dpi=240)
    plt.close(fig)


def build_figA7(records: list[dict]) -> None:
    reset_style()
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    })
    pixel = read_json(RESULTS / "similarity_rewriting" / "pixel_similarity.json")
    group_of = {(r["class_a"], r["class_b"]): r["group"] for r in records}
    sim = {pair: pixel[f"{pair[0]}-{pair[1]}"] for pair in group_of}
    pairs = sorted(group_of, key=lambda p: (group_of[p] != "similar", -sim[p], p))
    by_pair: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        by_pair[(r["condition"], (r["class_a"], r["class_b"]))].append(r["rewriting"])

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    width = 0.24
    x = np.arange(len(pairs), dtype=float)
    offsets = np.linspace(-width, width, num=len(CONDITIONS))
    for idx, (condition, label) in enumerate(CONDITIONS):
        vals = [float(np.mean(by_pair[(condition, p)])) for p in pairs]
        errs = [sd(by_pair[(condition, p)]) for p in pairs]
        ax.bar(x + offsets[idx], vals, width=width, yerr=errs, capsize=2.5, color=CONDITION_COLORS[condition],
               edgecolor="black", linewidth=0.5, error_kw={"linewidth": 0.9}, label=label, alpha=0.9)
    for i, pair in enumerate(pairs):
        if group_of[pair] == "dissimilar":
            ax.axvspan(i - 0.5, i + 0.5, color="#F2F2F2", zorder=0)
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\u2192{b}\n{group_of[(a, b)]}\npixel sim {sim[(a, b)]:.3f}" for a, b in pairs])
    ax.set_ylabel("Mean recurrent rewriting")
    ax.set_title("Recurrent E\u2192E rewriting by class pair", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=len(CONDITIONS))
    fig.tight_layout()
    save(fig, "figA7_pair_breakdown", dpi=240)
    plt.close(fig)


def main() -> None:
    records = load_records()
    summary = summarize(records)
    build_fig5(summary)
    build_figA7(records)
    for condition, label in CONDITIONS:
        s = summary[condition]
        print(f"{condition:<9} similar {s['similar']['mean']:.4f} ± {s['similar']['sd']:.4f} | "
              f"dissimilar {s['dissimilar']['mean']:.4f} ± {s['dissimilar']['sd']:.4f} | "
              f"ratio {s['ratio']:.2f}x | Mann-Whitney p={s['mann_whitney_p']:.2g} (n={s['similar']['n']} per group)")


if __name__ == "__main__":
    main()
