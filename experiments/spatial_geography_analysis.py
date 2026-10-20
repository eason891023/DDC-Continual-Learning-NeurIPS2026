"""
Two analyses that test the 'synaptic resource geography' claim:

Analysis 1 — Spatial visualization of per-task important synapse territories.
  Shows WHERE each task's important synapses live in 3D space, and whether
  different tasks occupy distinct spatial regions under DDC vs random.

Analysis 2 — Mask-level vs learning-level synapse span.
  Tests whether plasticity actually constrains the synaptic pool beyond what
  the DDC mask already imposes.  If mask_span ≈ learning_span, then Stage 1→2
  adds nothing; if learning_span < mask_span, plasticity does real work.

Usage:
    python -m experiments.spatial_geography_analysis \
        --runs_dir outputs/snapshot_runs/runs \
        --save_dir outputs/spatial_geography
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────
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

SIGMA_COLORS = {
    0.5: "#E41A1C",
    0.7: "#377EB8",
    1.0: "#4DAF4A",
    1.3: "#984EA3",
    2.0: "#FF7F00",
    "random": "#A65628",
}


# ── Helpers ────────────────────────────────────────────────────────────

def parse_run_dir(name: str):
    """Return (condition_label, sigma_or_none) from run directory name."""
    if name.startswith("random"):
        return "random", None
    m = re.match(r"distance_sigma([\d.]+)", name)
    if m:
        return f"sigma={m.group(1)}", float(m.group(1))
    return None, None


def load_task_snapshot(run_dir: Path, task_idx: int):
    """Load weights and neurons_labels for a given task snapshot."""
    snap_dir = run_dir / "task_snapshots" / f"task_{task_idx:02d}_class_{task_idx}"
    if not snap_dir.exists():
        return None
    ee_w = torch.load(snap_dir / "RSNN_EE_weights.pt", map_location="cpu", weights_only=True)
    nl = torch.load(snap_dir / "neurons_labels.pt", map_location="cpu", weights_only=True)
    inp_w = torch.load(snap_dir / "inputs2RSNN_EE_weights.pt", map_location="cpu", weights_only=True)
    return ee_w, nl, inp_w


def load_coords_and_mask(run_dir: Path):
    """Load neuron coordinates and EE mask."""
    ckpt = run_dir / "checkpoints" / "trained"
    coords = torch.load(ckpt / "RSNN_Exc_coords.pt", map_location="cpu", weights_only=True)
    mask = torch.load(ckpt / "RSNN_EE_mask.pt", map_location="cpu", weights_only=True)
    return coords.numpy(), mask


def compute_important_synapses_ee(ee_w, nl, class_idx, mask, top_k=4000):
    """
    Compute task-specific important EE synapses.
    importance = |w_ij| * a_i(c) * a_j(c)
    Returns indices (post_idx, pre_idx) of top-k synapses.
    """
    affinity = nl[:, class_idx].clamp(min=0)
    denom = affinity.sum()
    if denom > 0:
        affinity = affinity / denom

    # importance matrix: |w_ij| * a_i * a_j
    importance = ee_w.abs() * mask.float() * affinity.unsqueeze(1) * affinity.unsqueeze(0)

    # flatten and get top-k
    flat = importance.view(-1)
    _, top_indices = flat.topk(min(top_k, (flat > 0).sum().item()))
    post_idx = top_indices // importance.shape[1]
    pre_idx = top_indices % importance.shape[1]
    return post_idx.numpy(), pre_idx.numpy()


def synapse_span(coords, post_idx, pre_idx):
    """Mean Euclidean distance between pre and post neurons of synapse set."""
    if len(post_idx) == 0:
        return 0.0
    d = np.linalg.norm(coords[post_idx] - coords[pre_idx], axis=1)
    return float(d.mean())


def synapse_centroid(coords, post_idx, pre_idx):
    """Centroid of all neurons involved in a synapse set."""
    all_neurons = np.unique(np.concatenate([post_idx, pre_idx]))
    return coords[all_neurons].mean(axis=0)


def synapse_territory_neurons(post_idx, pre_idx):
    """Unique neurons involved in important synapses."""
    return set(np.unique(np.concatenate([post_idx, pre_idx])).tolist())


def mask_synapse_span(coords, mask, n_sample=4000):
    """Synapse span of randomly sampled connected synapses (mask-level baseline)."""
    connected = torch.nonzero(mask, as_tuple=False)
    n_conn = connected.shape[0]
    if n_conn == 0:
        return 0.0
    idx = torch.randperm(n_conn)[:n_sample]
    post_idx = connected[idx, 0].numpy()
    pre_idx = connected[idx, 1].numpy()
    return synapse_span(coords, post_idx, pre_idx)


# ── Analysis 1: Spatial Geography ──────────────────────────────────────

def analysis_1_geography(runs_dir: Path, save_dir: Path, sigmas_show=None):
    """
    For each condition, for each task:
      - compute important EE synapse set
      - compute centroid in 3D space
      - compute pairwise territory (neuron set) overlap

    Output:
      - 3D scatter of per-task centroids (one subplot per condition)
      - Pairwise territory Jaccard matrix (heatmap)
      - Bar plot of mean inter-task centroid distance
    """
    if sigmas_show is None:
        sigmas_show = [0.5, 0.7, 1.0, "random"]

    runs = sorted(runs_dir.iterdir())
    # Group runs by condition
    condition_runs = defaultdict(list)
    for rd in runs:
        if not rd.is_dir():
            continue
        label, sigma = parse_run_dir(rd.name)
        if label is None:
            continue
        if sigma is not None and sigma not in [s for s in sigmas_show if isinstance(s, (int, float))]:
            continue
        if label == "random" and "random" not in sigmas_show:
            continue
        condition_runs[label].append(rd)

    results = {}

    for label, run_dirs in sorted(condition_runs.items()):
        cond_centroids = []
        cond_territories = []
        cond_spans = []
        cond_mask_spans = []
        cond_inter_centroid = []
        cond_territory_jaccard = []

        for rd in run_dirs:
            try:
                coords, mask = load_coords_and_mask(rd)
            except Exception:
                continue

            task_centroids = []
            task_territories = []
            task_spans = []

            # mask-level span (Analysis 2 piggyback)
            ms = mask_synapse_span(coords, mask)
            cond_mask_spans.append(ms)

            for task_idx in range(10):
                snap = load_task_snapshot(rd, task_idx)
                if snap is None:
                    continue
                ee_w, nl, inp_w = snap
                post, pre = compute_important_synapses_ee(ee_w, nl, task_idx, mask)
                c = synapse_centroid(coords, post, pre)
                t = synapse_territory_neurons(post, pre)
                s = synapse_span(coords, post, pre)
                task_centroids.append(c)
                task_territories.append(t)
                task_spans.append(s)

            if len(task_centroids) < 10:
                continue

            cond_centroids.append(np.array(task_centroids))
            cond_territories.append(task_territories)
            cond_spans.append(np.mean(task_spans))

            # Inter-task centroid distances
            centroids_arr = np.array(task_centroids)
            dists = []
            for i in range(10):
                for j in range(i + 1, 10):
                    dists.append(np.linalg.norm(centroids_arr[i] - centroids_arr[j]))
            cond_inter_centroid.append(np.mean(dists))

            # Territory Jaccard matrix
            jaccard_mat = np.zeros((10, 10))
            for i in range(10):
                for j in range(10):
                    if i == j:
                        jaccard_mat[i, j] = 1.0
                    else:
                        inter = len(task_territories[i] & task_territories[j])
                        union = len(task_territories[i] | task_territories[j])
                        jaccard_mat[i, j] = inter / union if union > 0 else 0.0
            cond_territory_jaccard.append(jaccard_mat)

        if len(cond_centroids) == 0:
            continue

        results[label] = {
            "centroids": cond_centroids,
            "territories": cond_territories,
            "learning_span_mean": np.mean(cond_spans),
            "learning_span_std": np.std(cond_spans),
            "learning_span_values": cond_spans,
            "mask_span_mean": np.mean(cond_mask_spans),
            "mask_span_std": np.std(cond_mask_spans),
            "mask_span_values": cond_mask_spans,
            "inter_centroid_mean": np.mean(cond_inter_centroid),
            "inter_centroid_std": np.std(cond_inter_centroid),
            "inter_centroid_values": cond_inter_centroid,
            "territory_jaccard_matrices": cond_territory_jaccard,
            "territory_jaccard_mean": np.mean([m[np.triu_indices(10, k=1)] for m in cond_territory_jaccard]),
            "n_runs": len(cond_centroids),
        }

    # ── Plot 1: 3D centroid scatter (one panel per condition) ──
    n_conds = len(results)
    fig = plt.figure(figsize=(5 * n_conds, 5))
    task_colors = cm.tab10(np.linspace(0, 1, 10))

    for idx, (label, data) in enumerate(sorted(results.items())):
        ax = fig.add_subplot(1, n_conds, idx + 1, projection="3d")
        # Show centroids from the first seed
        centroids = data["centroids"][0]
        for t in range(10):
            ax.scatter(*centroids[t], color=task_colors[t], s=120, edgecolors="k",
                       linewidths=0.5, label=f"class {t}", zorder=5)
        ax.set_title(f"{label}\ninter-centroid = {data['inter_centroid_mean']:.3f}",
                     fontsize=11)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.suptitle("Spatial centroids of per-task important synapse sets", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "geography_3d_centroids.png", bbox_inches="tight")
    fig.savefig(save_dir / "geography_3d_centroids.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Territory Jaccard heatmaps ──
    fig, axes = plt.subplots(1, n_conds, figsize=(4.5 * n_conds, 4))
    if n_conds == 1:
        axes = [axes]
    for idx, (label, data) in enumerate(sorted(results.items())):
        ax = axes[idx]
        # Average Jaccard across seeds
        mean_jac = np.mean(data["territory_jaccard_matrices"], axis=0)
        im = ax.imshow(mean_jac, vmin=0, vmax=0.5, cmap="Reds")
        ax.set_title(f"{label}\nmean off-diag = {data['territory_jaccard_mean']:.3f}", fontsize=11)
        ax.set_xlabel("class")
        ax.set_ylabel("class")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
    fig.colorbar(im, ax=axes, label="Jaccard overlap", shrink=0.8)
    fig.suptitle("Per-task important-synapse territory overlap", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / "geography_territory_jaccard.png", bbox_inches="tight")
    fig.savefig(save_dir / "geography_territory_jaccard.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Inter-task centroid distance bar plot ──
    fig, ax = plt.subplots(figsize=(6, 4))
    labels_sorted = sorted(results.keys())
    means = [results[l]["inter_centroid_mean"] for l in labels_sorted]
    stds = [results[l]["inter_centroid_std"] for l in labels_sorted]
    x = np.arange(len(labels_sorted))
    colors = []
    for l in labels_sorted:
        if l == "random":
            colors.append(SIGMA_COLORS["random"])
        else:
            s = float(l.split("=")[1])
            colors.append(SIGMA_COLORS.get(s, "#999999"))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="k", linewidth=0.5,
           capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, rotation=30, ha="right")
    ax.set_ylabel("Mean inter-task centroid distance")
    ax.set_title("Spatial spread of task-specific synapse centroids")
    fig.tight_layout()
    fig.savefig(save_dir / "geography_centroid_distance.png", bbox_inches="tight")
    fig.savefig(save_dir / "geography_centroid_distance.pdf", bbox_inches="tight")
    plt.close(fig)

    return results


# ── Analysis 2: Mask vs Learning Synapse Span ──────────────────────────

def analysis_2_mask_vs_learning(results: dict, save_dir: Path):
    """
    Compare mask-level synapse span (random sample of connected synapses)
    against learning-level synapse span (task-important synapses).

    If mask_span ≈ learning_span: plasticity doesn't further constrain →
        Stage 2 is just DDC mask consequence
    If learning_span < mask_span: plasticity genuinely consolidates a
        more local pool → Stage 1→2 is a real causal link
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels_sorted = sorted(results.keys())
    x = np.arange(len(labels_sorted))
    width = 0.35

    mask_means = [results[l]["mask_span_mean"] for l in labels_sorted]
    mask_stds = [results[l]["mask_span_std"] for l in labels_sorted]
    learn_means = [results[l]["learning_span_mean"] for l in labels_sorted]
    learn_stds = [results[l]["learning_span_std"] for l in labels_sorted]

    ax.bar(x - width / 2, mask_means, width, yerr=mask_stds, label="Mask-level (random connected synapses)",
           color="#BDBDBD", edgecolor="k", linewidth=0.5, capsize=3)
    ax.bar(x + width / 2, learn_means, width, yerr=learn_stds, label="Learning-level (task-important synapses)",
           color="#2166AC", edgecolor="k", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, rotation=30, ha="right")
    ax.set_ylabel("Synapse span (mean pre-post distance)")
    ax.set_title("Does plasticity constrain the synaptic pool beyond the mask?")
    ax.legend(loc="upper left", framealpha=0.9)

    # Add delta annotations
    for i, l in enumerate(labels_sorted):
        delta = results[l]["mask_span_mean"] - results[l]["learning_span_mean"]
        pct = delta / results[l]["mask_span_mean"] * 100
        ax.annotate(f"Δ={delta:.3f}\n({pct:.1f}%)",
                    xy=(i, max(mask_means[i], learn_means[i]) + max(mask_stds[i], learn_stds[i]) + 0.01),
                    ha="center", fontsize=8, color="#333333")

    fig.tight_layout()
    fig.savefig(save_dir / "mask_vs_learning_span.png", bbox_inches="tight")
    fig.savefig(save_dir / "mask_vs_learning_span.pdf", bbox_inches="tight")
    plt.close(fig)

    # Statistical test per condition
    span_stats = {}
    for l in labels_sorted:
        mask_vals = results[l]["mask_span_values"]
        learn_vals = results[l]["learning_span_values"]
        n = min(len(mask_vals), len(learn_vals))
        if n >= 3:
            t_stat, p_val = stats.ttest_rel(mask_vals[:n], learn_vals[:n])
        else:
            t_stat, p_val = float("nan"), float("nan")
        span_stats[l] = {
            "mask_span": f"{results[l]['mask_span_mean']:.4f} ± {results[l]['mask_span_std']:.4f}",
            "learning_span": f"{results[l]['learning_span_mean']:.4f} ± {results[l]['learning_span_std']:.4f}",
            "delta": f"{results[l]['mask_span_mean'] - results[l]['learning_span_mean']:.4f}",
            "delta_pct": f"{(results[l]['mask_span_mean'] - results[l]['learning_span_mean']) / results[l]['mask_span_mean'] * 100:.1f}%",
            "paired_t_p": f"{p_val:.4f}" if not np.isnan(p_val) else "n/a",
            "n_runs": results[l]["n_runs"],
        }

    return span_stats


# ── Analysis 3: σ=0.5 bottleneck visualization ────────────────────────

def analysis_3_bottleneck_viz(runs_dir: Path, save_dir: Path):
    """
    For σ=0.5 and σ=0.7, show the spatial distribution of consecutive
    tasks' important synapse neurons. Demonstrate that σ=0.5 tasks are
    spatially separated at the neuron level but squeezed into the same
    narrow synaptic band.
    """
    conditions = {
        "sigma=0.5": [],
        "sigma=0.7": [],
        "random": [],
    }

    for rd in sorted(runs_dir.iterdir()):
        if not rd.is_dir():
            continue
        label, _ = parse_run_dir(rd.name)
        if label in conditions:
            conditions[label].append(rd)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Spatial distribution of important-synapse neurons for consecutive tasks\n"
                 "(task t in blue, task t+1 in red, overlap in purple)", fontsize=13)

    for row_idx, (label, run_dirs) in enumerate(
        [(k, conditions[k]) for k in ["sigma=0.5", "sigma=0.7", "random"]]
    ):
        if not run_dirs:
            continue
        rd = run_dirs[0]
        try:
            coords, mask = load_coords_and_mask(rd)
        except Exception:
            continue

        # Show 3 consecutive pairs: (0,1), (3,4), (6,7)
        pairs = [(0, 1), (3, 4), (6, 7)]
        for col_idx, (t_a, t_b) in enumerate(pairs):
            ax = axes[row_idx, col_idx]

            snap_a = load_task_snapshot(rd, t_a)
            snap_b = load_task_snapshot(rd, t_b)
            if snap_a is None or snap_b is None:
                ax.set_visible(False)
                continue

            ee_w_a, nl_a, _ = snap_a
            ee_w_b, nl_b, _ = snap_b

            post_a, pre_a = compute_important_synapses_ee(ee_w_a, nl_a, t_a, mask)
            post_b, pre_b = compute_important_synapses_ee(ee_w_b, nl_b, t_b, mask)

            neurons_a = set(np.unique(np.concatenate([post_a, pre_a])).tolist())
            neurons_b = set(np.unique(np.concatenate([post_b, pre_b])).tolist())
            overlap_neurons = neurons_a & neurons_b
            only_a = neurons_a - overlap_neurons
            only_b = neurons_b - overlap_neurons

            # Project to 2D (first two principal axes of coords)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(coords)

            # Plot all neurons as background
            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c="#E0E0E0", s=1, alpha=0.3, zorder=1)

            # Plot task-specific neurons
            if only_a:
                idx_a = list(only_a)
                ax.scatter(coords_2d[idx_a, 0], coords_2d[idx_a, 1],
                           c="#2166AC", s=8, alpha=0.6, zorder=2, label=f"class {t_a} only")
            if only_b:
                idx_b = list(only_b)
                ax.scatter(coords_2d[idx_b, 0], coords_2d[idx_b, 1],
                           c="#D6604D", s=8, alpha=0.6, zorder=2, label=f"class {t_b} only")
            if overlap_neurons:
                idx_o = list(overlap_neurons)
                ax.scatter(coords_2d[idx_o, 0], coords_2d[idx_o, 1],
                           c="#7B2D8E", s=12, alpha=0.8, zorder=3, label="shared")

            neuron_jaccard = len(overlap_neurons) / len(neurons_a | neurons_b) if (neurons_a | neurons_b) else 0
            # Compute synapse-level contention
            synapses_a = set(zip(post_a.tolist(), pre_a.tolist()))
            synapses_b = set(zip(post_b.tolist(), pre_b.tolist()))
            syn_jaccard = len(synapses_a & synapses_b) / len(synapses_a | synapses_b) if (synapses_a | synapses_b) else 0

            ax.set_title(f"{label}: class {t_a} vs {t_b}\n"
                         f"neuron J={neuron_jaccard:.3f}, synapse J={syn_jaccard:.4f}",
                         fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="lower right")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_dir / "geography_bottleneck_comparison.png", bbox_inches="tight")
    fig.savefig(save_dir / "geography_bottleneck_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str,
                        default="outputs/snapshot_runs/runs")
    parser.add_argument("--save_dir", type=str,
                        default="outputs/spatial_geography")
    parser.add_argument("--sigmas", type=float, nargs="*", default=[0.5, 0.7, 1.0])
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sigmas_show = args.sigmas + ["random"]

    print("=" * 60)
    print("Analysis 1: Spatial Geography of Per-Task Important Synapses")
    print("=" * 60)
    results = analysis_1_geography(runs_dir, save_dir, sigmas_show=sigmas_show)

    print("\n" + "=" * 60)
    print("Analysis 2: Mask vs Learning Synapse Span")
    print("=" * 60)
    span_stats = analysis_2_mask_vs_learning(results, save_dir)

    for label, st in sorted(span_stats.items()):
        print(f"\n  {label}:")
        for k, v in st.items():
            print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("Analysis 3: σ=0.5 vs σ=0.7 Bottleneck Visualization")
    print("=" * 60)
    analysis_3_bottleneck_viz(runs_dir, save_dir)

    # Save summary JSON
    summary = {
        "analysis_1_geography": {},
        "analysis_2_mask_vs_learning": span_stats,
    }
    for label, data in results.items():
        summary["analysis_1_geography"][label] = {
            "inter_centroid_mean": data["inter_centroid_mean"],
            "inter_centroid_std": data["inter_centroid_std"],
            "inter_centroid_values": data["inter_centroid_values"],
            "territory_jaccard_mean": float(data["territory_jaccard_mean"]),
            "learning_span_mean": data["learning_span_mean"],
            "learning_span_std": data["learning_span_std"],
            "mask_span_mean": data["mask_span_mean"],
            "mask_span_std": data["mask_span_std"],
            "n_runs": data["n_runs"],
        }

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nAll outputs saved to {save_dir}/")
    print("Key files:")
    print(f"  geography_3d_centroids.png — per-task spatial centroids")
    print(f"  geography_territory_jaccard.png — neuron-territory overlap heatmaps")
    print(f"  geography_centroid_distance.png — inter-task centroid spread")
    print(f"  mask_vs_learning_span.png — does plasticity constrain beyond mask?")
    print(f"  geography_bottleneck_comparison.png — σ=0.5 vs 0.7 spatial comparison")
    print(f"  summary.json — all numbers")


if __name__ == "__main__":
    main()
