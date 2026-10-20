"""Figure 2: DDC shapes synaptic resource geography.

(a) structural span, (b) structural versus learned span, (c) consecutive-task contention,
(d) distance-resolved contention. Spans and pooled contention were stored as mean and
population SD over n runs; they are converted to sample SD here (see results/mechanism).

Input:  results/mechanism/fig2_data.json
Output: figures/fig2_mechanism.{png,pdf}

Usage: python -m paper_figures.fig2_mechanism
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_figures.common import FIG_DIR, RESULTS, read_json, reset_style, sd, sd_from_population

COLORS = {
    "sigma=0.4": "#D7301F",
    "sigma=0.5": "#E41A1C",
    "sigma=0.7": "#377EB8",
    "sigma=1.0": "#4DAF4A",
    "sigma=2.0": "#FF7F00",
    "random": "#A65628",
}
C_GRID = "#D9D9D9"
C_MASK = "#BDBDBD"
C_LEARN = "#2166AC"

STYLE = {
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 180,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load() -> dict:
    data = read_json(RESULTS / "mechanism" / "fig2_data.json")
    span = {}
    for key, s in data["span"].items():
        n = s["n_runs"]
        span[key] = {
            "mask_mean": s["structural_span_mean"],
            "mask_sd": sd_from_population(s["structural_span_sd_population"], n),
            "learn_mean": s["learned_span_mean"],
            "learn_sd": sd_from_population(s["learned_span_sd_population"], n),
        }
    contention = {}
    for key, c in data["consecutive_contention_ee"].items():
        spread = sd(c["values"]) if "values" in c else sd_from_population(c["sd_population"], c["n_runs"])
        contention[key] = {"mean": c["mean"], "sd": spread}
    prof = data["distance_resolved_contention_ee"]
    edges = np.array(prof["bin_edges"])
    profiles = {}
    for key, runs in prof["runs"].items():
        arr = np.array([r["profile"] for r in runs])
        profiles[key] = {"mean": arr.mean(axis=0), "sd": np.array([sd(col) for col in arr.T]), "n": len(runs)}
    return {"span": span, "contention": contention, "edges": edges, "profiles": profiles}


def build() -> dict:
    reset_style()
    plt.rcParams.update(STYLE)
    d = load()
    geography, pool, profiles = d["span"], d["contention"], d["profiles"]
    bin_centers = (d["edges"][:-1] + d["edges"][1:]) / 2

    order_main = ["sigma=0.4", "sigma=0.7", "sigma=1.0", "random"]
    order_tradeoff = ["sigma=0.4", "sigma=0.7", "sigma=1.0", "sigma=2.0", "random"]

    fig = plt.figure(figsize=(10.4, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], height_ratios=[1.0, 1.08], wspace=0.30, hspace=0.38)

    # (a) Structural pool geometry
    ax1 = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(order_main))
    mask_means = [geography[k]["mask_mean"] for k in order_main]
    mask_stds = [geography[k]["mask_sd"] for k in order_main]
    labels = [r"$\sigma=0.4$", r"$\sigma=0.7$", r"$\sigma=1.0$", "Random"]
    ax1.bar(xs, mask_means, yerr=mask_stds, color=[COLORS[k] for k in order_main],
            edgecolor="black", linewidth=0.7, capsize=4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean pre-post distance")
    ax1.set_title("(a) DDC constrains the\navailable synaptic pool", fontweight="bold", fontsize=16)
    ax1.grid(True, axis="y", color=C_GRID, alpha=0.8)
    ax1.set_ylim(0.40, 1.08)

    # (b) Structural versus learned span
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.34
    x2 = np.arange(len(order_main))
    learn_means = [geography[k]["learn_mean"] for k in order_main]
    learn_stds = [geography[k]["learn_sd"] for k in order_main]
    ax2.bar(x2 - width / 2, mask_means, width, yerr=mask_stds, color=C_MASK,
            edgecolor="black", linewidth=0.6, capsize=3, label="Structural")
    ax2.bar(x2 + width / 2, learn_means, width, yerr=learn_stds, color=C_LEARN,
            edgecolor="black", linewidth=0.6, capsize=3, label="Learned")
    for i, key in enumerate(order_main):
        pct = 100 * (geography[key]["mask_mean"] - geography[key]["learn_mean"]) / geography[key]["mask_mean"]
        ax2.text(i, max(mask_means[i], learn_means[i]) + 0.03, f"-{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean pre-post distance")
    ax2.set_title("(b) Plasticity further compresses\nthe structural pool", fontweight="bold", fontsize=16)
    ax2.grid(True, axis="y", color=C_GRID, alpha=0.8)
    ax2.legend(frameon=False, loc="upper left", handlelength=1.8)
    ax2.set_ylim(0.40, 1.08)

    # (c) Consecutive-task contention
    ax3 = fig.add_subplot(gs[1, 0])
    x3 = np.arange(len(order_tradeoff))
    contention_means = [pool[k]["mean"] * 100 for k in order_tradeoff]
    contention_stds = [pool[k]["sd"] * 100 for k in order_tradeoff]
    labels3 = [r"$\sigma=0.4$", r"$\sigma=0.7$", r"$\sigma=1.0$", r"$\sigma=2.0$", "Random"]
    ax3.bar(x3, contention_means, yerr=contention_stds, color=[COLORS[k] for k in order_tradeoff],
            edgecolor="black", linewidth=0.7, capsize=4)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels3)
    ax3.set_ylabel("Consecutive contention (%)")
    ax3.set_title("(c) Small-$\\sigma$ and random regimes\nboth show elevated contention", fontweight="bold", fontsize=16)
    ax3.grid(True, axis="y", color=C_GRID, alpha=0.8)
    ax3.set_ylim(0, max(contention_means) + max(contention_stds) + 1.5)

    # (d) Distance-resolved contention
    ax4 = fig.add_subplot(gs[1, 1])
    line_labels = {"sigma=0.4": r"$\sigma=0.4$", "sigma=0.7": r"$\sigma=0.7$", "random": "Random"}
    for key in ["sigma=0.4", "sigma=0.7", "random"]:
        mean_profile, sd_profile = profiles[key]["mean"], profiles[key]["sd"]
        color = COLORS[key]
        ax4.plot(bin_centers, mean_profile, color=color, linewidth=2.6, label=line_labels[key])
        ax4.fill_between(bin_centers, np.maximum(0.0, mean_profile - sd_profile), mean_profile + sd_profile,
                         color=color, alpha=0.12)
    ax4.set_xlabel("Synapse spatial distance (pre-post Euclidean)")
    ax4.set_ylabel("# shared important synapses\n(per consecutive pair)")
    ax4.set_title("(d) Similar contention magnitude,\nbut different spatial anatomy", fontweight="bold", fontsize=16)
    ax4.grid(True, axis="y", color=C_GRID, alpha=0.8)
    ax4.legend(frameon=False, loc="upper right")
    ax4.annotate("local bottleneck", xy=(0.36, profiles["sigma=0.4"]["mean"].max()), xytext=(0.63, 118),
                 textcoords="data", arrowprops=dict(arrowstyle="->", color=COLORS["sigma=0.4"], lw=1.5),
                 color=COLORS["sigma=0.4"], fontsize=11, fontweight="bold")
    ax4.annotate("diffuse contention", xy=(0.78, profiles["random"]["mean"][7]), xytext=(0.92, 47),
                 textcoords="data", arrowprops=dict(arrowstyle="->", color=COLORS["random"], lw=1.5),
                 color=COLORS["random"], fontsize=11, fontweight="bold")
    ax4.set_xlim(0.03, 1.48)
    ax4.set_ylim(0, max(profiles["sigma=0.4"]["mean"].max(), profiles["random"]["mean"].max()) + 18)

    fig.subplots_adjust(top=0.965, left=0.07, right=0.985, bottom=0.085)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig2_mechanism.png", bbox_inches="tight", dpi=240)
    fig.savefig(FIG_DIR / "fig2_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    return d


def main() -> None:
    d = build()
    print("Span (mean ± SD):")
    for key, s in d["span"].items():
        reduction = 100 * (s["mask_mean"] - s["learn_mean"]) / s["mask_mean"]
        print(f"  {key:<10} structural {s['mask_mean']:.3f} ± {s['mask_sd']:.3f}   "
              f"learned {s['learn_mean']:.3f} ± {s['learn_sd']:.3f}   reduction {reduction:.1f}%")
    print("Consecutive contention, E->E (mean ± SD):")
    for key, c in d["contention"].items():
        print(f"  {key:<10} {c['mean']:.4f} ± {c['sd']:.4f}")


if __name__ == "__main__":
    main()
