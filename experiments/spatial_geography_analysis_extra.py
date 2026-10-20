"""
Extra analyses building on spatial_geography_analysis:

Analysis 4 — Effective pool size and cross-task synapse contention by spatial zone.
  For σ=0.5 bottleneck claim: show that the AVAILABLE synaptic pool (connected
  synapses in the mask) per spatial region is much smaller, forcing different
  tasks to compete for the same synapses even when their neuron assemblies
  are separated.

Analysis 5 — Synapse contention decomposed by spatial distance.
  For each pair of consecutive tasks, measure how much of the contention
  (shared important synapses) comes from local vs distant synapses.
  Under σ=0.5, contention should be concentrated in local synapses.

Usage:
    python -m experiments.spatial_geography_analysis_extra
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from experiments.spatial_geography_analysis import (
    parse_run_dir,
    load_task_snapshot,
    load_coords_and_mask,
    compute_important_synapses_ee,
    SIGMA_COLORS,
)

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


def analysis_4_pool_size(runs_dir: Path, save_dir: Path):
    """
    For each condition, compute:
      - Total available synapses in mask (total pool)
      - Mean #unique synapses per task's important set
      - Cross-task synapse contention: fraction of task A's important synapses
        that also appear in task B's important set, for consecutive pairs
      - The ratio: important_synapses / available_pool → "pool utilization"
        Higher utilization = more competition = bottleneck
    """
    conditions_to_check = [0.5, 0.7, 1.0, 2.0]

    condition_data = defaultdict(lambda: {
        "total_pool": [],
        "per_task_important_unique": [],
        "consecutive_contention": [],
        "pool_utilization": [],
        "per_task_spans": [],
    })

    for rd in sorted(runs_dir.iterdir()):
        if not rd.is_dir():
            continue
        label, sigma = parse_run_dir(rd.name)
        if label is None:
            continue
        if sigma is not None and sigma not in conditions_to_check:
            continue
        if label != "random" and sigma not in conditions_to_check:
            continue

        try:
            coords, mask = load_coords_and_mask(rd)
        except Exception:
            continue

        total_connected = int(mask.sum().item())
        condition_data[label]["total_pool"].append(total_connected)

        task_synapse_sets = []
        task_unique_counts = []

        for task_idx in range(10):
            snap = load_task_snapshot(rd, task_idx)
            if snap is None:
                continue
            ee_w, nl, inp_w = snap
            post, pre = compute_important_synapses_ee(ee_w, nl, task_idx, mask)
            syn_set = set(zip(post.tolist(), pre.tolist()))
            task_synapse_sets.append(syn_set)
            task_unique_counts.append(len(syn_set))

            # span
            if len(post) > 0:
                d = np.linalg.norm(coords[post] - coords[pre], axis=1)
                condition_data[label]["per_task_spans"].append(float(d.mean()))

        if len(task_synapse_sets) < 10:
            continue

        condition_data[label]["per_task_important_unique"].append(np.mean(task_unique_counts))

        # Pool utilization: union of all tasks' important synapses / total pool
        all_important = set()
        for s in task_synapse_sets:
            all_important |= s
        condition_data[label]["pool_utilization"].append(len(all_important) / total_connected)

        # Consecutive contention
        contentions = []
        for t in range(9):
            inter = len(task_synapse_sets[t] & task_synapse_sets[t + 1])
            union = len(task_synapse_sets[t] | task_synapse_sets[t + 1])
            contentions.append(inter / union if union > 0 else 0)
        condition_data[label]["consecutive_contention"].append(np.mean(contentions))

    # ── Plot: Pool utilization + contention grouped bar ──
    labels = sorted(condition_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Total pool size
    ax = axes[0]
    means = [np.mean(condition_data[l]["total_pool"]) for l in labels]
    stds = [np.std(condition_data[l]["total_pool"]) for l in labels]
    colors = [SIGMA_COLORS.get("random" if l == "random" else float(l.split("=")[1]), "#999")
              for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, color=colors, edgecolor="k",
           linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Total connected synapses (EE)")
    ax.set_title("(a) Available synaptic pool size")

    # Panel B: Pool utilization
    ax = axes[1]
    means = [np.mean(condition_data[l]["pool_utilization"]) * 100 for l in labels]
    stds = [np.std(condition_data[l]["pool_utilization"]) * 100 for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, color=colors, edgecolor="k",
           linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Pool utilization (%)")
    ax.set_title("(b) Fraction of pool used by all 10 tasks")

    # Panel C: Consecutive contention
    ax = axes[2]
    means = [np.mean(condition_data[l]["consecutive_contention"]) * 100 for l in labels]
    stds = [np.std(condition_data[l]["consecutive_contention"]) * 100 for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, color=colors, edgecolor="k",
           linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Consecutive contention (Jaccard %)")
    ax.set_title("(c) Task-to-task synaptic overlap")

    fig.suptitle("Synaptic pool economics: smaller pool → higher utilization → more contention",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "pool_economics.png", bbox_inches="tight")
    fig.savefig(save_dir / "pool_economics.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Summary stats ──
    summary = {}
    for l in labels:
        cd = condition_data[l]
        summary[l] = {
            "total_pool_mean": float(np.mean(cd["total_pool"])),
            "pool_utilization_mean": float(np.mean(cd["pool_utilization"])),
            "pool_utilization_std": float(np.std(cd["pool_utilization"])),
            "consecutive_contention_mean": float(np.mean(cd["consecutive_contention"])),
            "consecutive_contention_std": float(np.std(cd["consecutive_contention"])),
            "n_runs": len(cd["total_pool"]),
        }
        print(f"\n  {l}:")
        print(f"    total pool: {summary[l]['total_pool_mean']:.0f}")
        print(f"    pool utilization: {summary[l]['pool_utilization_mean']*100:.1f}%")
        print(f"    consecutive contention: {summary[l]['consecutive_contention_mean']*100:.2f}%")

    with open(save_dir / "pool_economics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def analysis_5_contention_by_distance(runs_dir: Path, save_dir: Path):
    """
    For each pair of consecutive tasks, decompose the shared important
    synapses by the spatial distance of the synapse (pre-post distance).
    Show that under σ=0.5, contention is concentrated in short-range synapses.
    """
    conditions = {"sigma=0.5": 0.5, "sigma=0.7": 0.7, "sigma=1.0": 1.0, "random": None}
    dist_bins = np.linspace(0, 1.5, 16)
    bin_centers = (dist_bins[:-1] + dist_bins[1:]) / 2

    condition_profiles = defaultdict(list)

    for rd in sorted(runs_dir.iterdir()):
        if not rd.is_dir():
            continue
        label, sigma = parse_run_dir(rd.name)
        if label not in conditions:
            continue

        try:
            coords, mask = load_coords_and_mask(rd)
        except Exception:
            continue

        task_syn_with_dist = []
        for task_idx in range(10):
            snap = load_task_snapshot(rd, task_idx)
            if snap is None:
                continue
            ee_w, nl, _ = snap
            post, pre = compute_important_synapses_ee(ee_w, nl, task_idx, mask)
            # Store synapse set with distances
            dists = np.linalg.norm(coords[post] - coords[pre], axis=1)
            syn_dict = {}
            for i in range(len(post)):
                syn_dict[(post[i], pre[i])] = dists[i]
            task_syn_with_dist.append(syn_dict)

        if len(task_syn_with_dist) < 10:
            continue

        # For consecutive pairs, find shared synapses and their distances
        run_profile = np.zeros(len(bin_centers))
        n_pairs = 0
        for t in range(9):
            shared_keys = set(task_syn_with_dist[t].keys()) & set(task_syn_with_dist[t + 1].keys())
            if not shared_keys:
                continue
            shared_dists = [task_syn_with_dist[t][k] for k in shared_keys]
            hist, _ = np.histogram(shared_dists, bins=dist_bins)
            run_profile += hist
            n_pairs += 1

        if n_pairs > 0:
            run_profile /= n_pairs  # average across pairs
            condition_profiles[label].append(run_profile)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in ["sigma=0.5", "sigma=0.7", "sigma=1.0", "random"]:
        if label not in condition_profiles:
            continue
        profiles = np.array(condition_profiles[label])
        mean_profile = profiles.mean(axis=0)
        std_profile = profiles.std(axis=0)
        sigma_val = conditions[label]
        color = SIGMA_COLORS.get(sigma_val if sigma_val else "random", "#999")
        ax.plot(bin_centers, mean_profile, color=color, linewidth=2, label=label)
        ax.fill_between(bin_centers, mean_profile - std_profile, mean_profile + std_profile,
                        color=color, alpha=0.15)

    ax.set_xlabel("Synapse spatial distance (pre-post Euclidean)")
    ax.set_ylabel("# shared important synapses (per consecutive pair)")
    ax.set_title("Where does synaptic contention happen spatially?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "contention_by_distance.png", bbox_inches="tight")
    fig.savefig(save_dir / "contention_by_distance.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\nContention-by-distance profiles saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pool economics and distance-resolved contention")
    parser.add_argument("--runs_dir", type=Path, default=Path("outputs/snapshot_runs/runs"))
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/spatial_geography"))
    cli = parser.parse_args()
    runs_dir, save_dir = cli.runs_dir, cli.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 4: Pool Economics")
    print("=" * 60)
    analysis_4_pool_size(runs_dir, save_dir)

    print("\n" + "=" * 60)
    print("Analysis 5: Contention by Spatial Distance")
    print("=" * 60)
    analysis_5_contention_by_distance(runs_dir, save_dir)
