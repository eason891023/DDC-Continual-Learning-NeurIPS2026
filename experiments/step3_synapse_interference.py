"""
Synapse interference measurement for a two-class (A -> B) protocol.

The helper functions here are used by step3_functional_similarity_interference.py,
which implements the pairwise rewriting assay reported in the paper.

Goal: Bridge the gap between "separated assemblies" and "less forgetting"
      by directly measuring whether distance-dependent connectivity reduces
      weight interference between sequentially learned classes.

Protocol:
  For each (mode, seed) combination:
    1. Build model, burn-in → save W0 (initial weights)
    2. Train class A → save W_A
    3. Train class B → save W_B
    4. Compute:
       - ΔW_A = W_A - W0  (weight changes from learning A)
       - ΔW_B = W_B - W_A  (weight changes from learning B)
    5. Measure interference:
       - Cosine similarity between |ΔW_A| and |ΔW_B|
       - "Important synapse overlap": how much does B change A's important synapses?
       - Spatial overlap of modified synapse sets

  Test on class pairs:
    - Similar pairs (high pixel overlap): (3,5), (5,8), (3,8)
    - Dissimilar pairs (low pixel overlap): (0,1), (1,4), (0,7)

Prediction: Distance network has lower synapse interference than random network,
            especially for similar class pairs.

Usage:
  python -m experiments.step3_synapse_interference --device cuda --n_seeds 3
"""

import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import os
import json
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main.Network import Brain_Like_RSNN


# ─────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_per_class_subsets(dataset, per_class=10, seed=42):
    np.random.seed(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    subsets = {}
    for label in range(10):
        indices = class_indices[label]
        np.random.shuffle(indices)
        subsets[label] = Subset(dataset, indices[:per_class])
    return subsets


def snapshot_weights(model):
    """Return a dict of cloned weight tensors."""
    return {
        'input_EE': model.inputs2RSNN_EE_synapase.inputs_EE_weights.clone(),
        'RSNN_EE': model.RSNN_EE_synapase.RSNN_EE_weights.clone(),
        'RSNN_IE': model.RSNN_IE_synapase.IE_weights.clone(),
    }


def compute_delta(w_after, w_before):
    """Compute weight change dict."""
    return {k: w_after[k] - w_before[k] for k in w_after}


# ─────────────────────────────────────────────────
#  Interference metrics
# ─────────────────────────────────────────────────

def compute_interference_metrics(delta_A, delta_B, mask_dict=None):
    """
    Compute interference metrics between weight changes from class A and class B.

    Args:
        delta_A: dict of ΔW tensors from learning class A
        delta_B: dict of ΔW tensors from learning class B
        mask_dict: optional dict of masks (only measure within connected synapses)

    Returns:
        dict of metrics per weight matrix and aggregated
    """
    results = {}

    for key in delta_A:
        dA = delta_A[key].flatten().float()
        dB = delta_B[key].flatten().float()

        # Apply mask if available (only measure connected synapses)
        if mask_dict and key in mask_dict:
            mask = mask_dict[key].flatten().bool()
            dA = dA[mask]
            dB = dB[mask]

        abs_dA = dA.abs()
        abs_dB = dB.abs()

        # 1. Cosine similarity of absolute weight changes
        #    High = same synapses are being modified by both classes
        cos_sim = torch.nn.functional.cosine_similarity(
            abs_dA.unsqueeze(0), abs_dB.unsqueeze(0)
        ).item()

        # 2. Important synapse overlap
        #    "Important" for A = top 5% most changed synapses
        #    Measure: mean |ΔW_B| at A-important synapses vs mean |ΔW_B| elsewhere
        k = max(1, int(0.05 * len(dA)))
        topk_A = torch.topk(abs_dA, k).indices
        mask_important = torch.zeros_like(abs_dA, dtype=torch.bool)
        mask_important[topk_A] = True

        mean_dB_at_important = abs_dB[mask_important].mean().item()
        mean_dB_elsewhere = abs_dB[~mask_important].mean().item()
        importance_ratio = mean_dB_at_important / (mean_dB_elsewhere + 1e-10)

        # 3. Overlap of top-k modified synapses
        #    Both A and B have their own top-5% → Jaccard index
        topk_B = torch.topk(abs_dB, k).indices
        set_A = set(topk_A.cpu().numpy().tolist())
        set_B = set(topk_B.cpu().numpy().tolist())
        jaccard = len(set_A & set_B) / len(set_A | set_B) if len(set_A | set_B) > 0 else 0

        # 4. Directional interference
        #    For A-important synapses, does B push them in the OPPOSITE direction?
        #    (negative dot product = destructive interference)
        dot_at_important = (dA[mask_important] * dB[mask_important]).mean().item()

        results[key] = {
            'cosine_similarity': cos_sim,
            'importance_ratio': importance_ratio,
            'mean_dB_at_A_important': mean_dB_at_important,
            'mean_dB_elsewhere': mean_dB_elsewhere,
            'topk_jaccard': jaccard,
            'directional_interference': dot_at_important,
            'mean_abs_dA': abs_dA.mean().item(),
            'mean_abs_dB': abs_dB.mean().item(),
        }

    return results


# ─────────────────────────────────────────────────
#  Training helper
# ─────────────────────────────────────────────────

def train_one_class(model, loader, cls, T_on, T_off, T_skip, device):
    """Train model on one class, return neurons_labels column."""
    model.enable_plasticity()
    spike_counts = torch.zeros(model.exc_neurons, device=device)

    with torch.no_grad():
        for img, label in tqdm(loader, desc=f"  Train cls {cls}", leave=False):
            x = img.to(device)

            for t in range(T_on):
                exc_spikes, _ = model(x)
                model.step()
                if t >= T_skip:
                    spike_counts += exc_spikes[0]

            zero_input = torch.zeros_like(x)
            for t in range(T_off):
                exc_spikes, _ = model(zero_input)
                model.step()
                spike_counts += exc_spikes[0]

    return spike_counts


# ─────────────────────────────────────────────────
#  Single run (one mode × one class pair × one seed)
# ─────────────────────────────────────────────────

def run_single(args, mode, sigma, seed, cls_A, cls_B, mnist_train):
    device = args.device
    set_seed(seed)

    train_subsets = get_per_class_subsets(mnist_train, per_class=args.per_class, seed=seed)

    # Build model
    set_seed(seed)
    model = Brain_Like_RSNN(
        inputs_size=torch.Size([1, 1, 28, 28]),
        exc_neurons=4096, inh_neurons=1024,
        sigma=sigma, dt=0.1, mode=mode, input_mode='spatial',
        target_mean=args.target_mean, device=device
    ).to(device)

    sample_img, _ = train_subsets[0][0]
    model.build(sample_img.unsqueeze(0).to(device))

    # Burn-in
    all_train_indices = []
    for cls in range(10):
        all_train_indices.extend(train_subsets[cls].indices)
    burn_loader = DataLoader(Subset(mnist_train, all_train_indices),
                              batch_size=1, shuffle=False)
    model.burn_in(burn_loader, T_on=args.burn_in_on, T_off=args.burn_in_off)

    # Get masks for filtering
    mask_dict = {
        'input_EE': model.inputs2RSNN_EE_synapase.inputs_EE_mask,
        'RSNN_EE': model.RSNN_EE_synapase.RSNN_EE_mask,
        'RSNN_IE': model.RSNN_IE_synapase.IE_weights_mask,
    }

    # Step 1: Snapshot initial weights
    W0 = snapshot_weights(model)

    # Step 2: Train class A
    loader_A = DataLoader(train_subsets[cls_A], batch_size=1, shuffle=False)
    train_one_class(model, loader_A, cls_A, args.T_on, args.T_off, args.T_skip, device)
    W_A = snapshot_weights(model)

    # Step 3: Train class B
    loader_B = DataLoader(train_subsets[cls_B], batch_size=1, shuffle=False)
    train_one_class(model, loader_B, cls_B, args.T_on, args.T_off, args.T_skip, device)
    W_B = snapshot_weights(model)

    # Step 4: Compute deltas
    delta_A = compute_delta(W_A, W0)
    delta_B = compute_delta(W_B, W_A)

    # Step 5: Compute interference
    metrics = compute_interference_metrics(delta_A, delta_B, mask_dict)

    del model
    if device != 'cpu':
        torch.cuda.empty_cache()

    return metrics


# ─────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────

def plot_interference_comparison(all_results, save_dir):
    """Bar plot comparing interference metrics: distance vs random, similar vs dissimilar pairs."""
    import matplotlib.pyplot as plt

    # Organize by (mode, pair_type)
    # pair_type: 'similar' or 'dissimilar'
    similar_pairs = [(3, 5), (5, 8), (3, 8)]
    dissimilar_pairs = [(0, 1), (1, 4), (0, 7)]

    def get_pair_type(cls_A, cls_B):
        if (cls_A, cls_B) in similar_pairs or (cls_B, cls_A) in similar_pairs:
            return 'similar'
        return 'dissimilar'

    # Aggregate metrics
    weight_keys = ['input_EE', 'RSNN_EE', 'RSNN_IE']
    metric_names = ['cosine_similarity', 'importance_ratio', 'topk_jaccard']
    metric_labels = ['Cosine Similarity\n(|ΔW_A| vs |ΔW_B|)',
                     'Importance Ratio\n(ΔW_B at A-important / elsewhere)',
                     'Top-5% Synapse\nJaccard Overlap']

    for wk in weight_keys:
        fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))

        for ax, mn, ml in zip(axes, metric_names, metric_labels):
            # Collect values: 4 groups (dist-sim, dist-dissim, rand-sim, rand-dissim)
            groups = {
                'dist\nsimilar': [], 'dist\ndissimilar': [],
                'rand\nsimilar': [], 'rand\ndissimilar': []
            }

            for r in all_results:
                pt = get_pair_type(r['cls_A'], r['cls_B'])
                mode = r['mode']
                val = r['metrics'][wk][mn]
                key = f"{'dist' if mode == 'distance' else 'rand'}\n{pt}"
                groups[key].append(val)

            names = list(groups.keys())
            means = [np.mean(groups[k]) if groups[k] else 0 for k in names]
            stds = [np.std(groups[k]) if groups[k] else 0 for k in names]

            colors = ['#2196F3', '#64B5F6', '#F44336', '#EF9A9A']
            bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                         edgecolor='black', linewidth=0.5)

            # Scatter individual points
            for i, k in enumerate(names):
                if groups[k]:
                    ax.scatter([i] * len(groups[k]), groups[k],
                              color='black', s=15, zorder=5, alpha=0.6)

            ax.set_ylabel(ml)
            ax.set_title(mn)

        fig.suptitle(f'Synapse Interference — {wk} weights', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'interference_{wk}.png'), dpi=200, bbox_inches='tight')
        plt.close()


def plot_directional_interference(all_results, save_dir):
    """Plot directional interference: does B push A-important synapses in opposite direction?"""
    similar_pairs = [(3, 5), (5, 8), (3, 8)]

    def get_pair_type(cls_A, cls_B):
        if (cls_A, cls_B) in similar_pairs or (cls_B, cls_A) in similar_pairs:
            return 'similar'
        return 'dissimilar'

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    weight_keys = ['input_EE', 'RSNN_EE', 'RSNN_IE']

    for ax, wk in zip(axes, weight_keys):
        groups = {
            'dist-sim': [], 'dist-dissim': [],
            'rand-sim': [], 'rand-dissim': []
        }

        for r in all_results:
            pt = get_pair_type(r['cls_A'], r['cls_B'])
            mode = 'dist' if r['mode'] == 'distance' else 'rand'
            val = r['metrics'][wk]['directional_interference']
            groups[f'{mode}-{pt[:3]}'] = groups.get(f'{mode}-{pt[:3]}', [])
            key = f"{mode}-{'sim' if pt == 'similar' else 'dissim'}"
            groups[key].append(val)

        names = list(groups.keys())
        means = [np.mean(groups[k]) if groups[k] else 0 for k in names]
        stds = [np.std(groups[k]) if groups[k] else 0 for k in names]

        colors = ['#2196F3', '#64B5F6', '#F44336', '#EF9A9A']
        ax.bar(names, means, yerr=stds, capsize=5, color=colors,
               edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Directional Interference\n(negative = destructive)')
        ax.set_title(wk)

    fig.suptitle('Directional Interference at A-important Synapses', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'directional_interference.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────
#  Main experiment
# ─────────────────────────────────────────────────

def run_experiment(args):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Class pairs to test
    similar_pairs = [(3, 5), (5, 8), (3, 8)]
    dissimilar_pairs = [(0, 1), (1, 4), (0, 7)]
    all_pairs = similar_pairs + dissimilar_pairs

    # Conditions
    conditions = [
        ('distance', args.sigma),
        ('random', 1.0),
    ]

    all_results = []
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    total_runs = len(conditions) * len(all_pairs) * len(seeds)
    run_idx = 0

    for mode, sigma in conditions:
        for cls_A, cls_B in all_pairs:
            for seed in seeds:
                run_idx += 1
                cond_name = f"{mode}_sigma{sigma}" if mode == 'distance' else 'random'
                print(f"\n[{run_idx}/{total_runs}] {cond_name} | pair ({cls_A},{cls_B}) | seed {seed}")

                metrics = run_single(args, mode, sigma, seed, cls_A, cls_B, mnist_train)

                result = {
                    'mode': mode,
                    'sigma': sigma,
                    'cls_A': cls_A,
                    'cls_B': cls_B,
                    'seed': seed,
                    'metrics': metrics,
                }
                all_results.append(result)

                # Save incrementally
                with open(os.path.join(save_dir, 'all_results.json'), 'w') as f:
                    json.dump(all_results, f, indent=2)

    # ===== Aggregate and print summary =====
    print(f"\n{'='*80}")
    print(f"  SYNAPSE INTERFERENCE RESULTS (N={args.n_seeds} seeds)")
    print(f"{'='*80}")

    for wk in ['input_EE', 'RSNN_EE', 'RSNN_IE']:
        print(f"\n  --- {wk} weights ---")
        print(f"  {'Condition':<15} {'Pair Type':<12} {'Cosine Sim':>12} {'Imp. Ratio':>12} {'Jaccard':>10} {'Direction':>12}")
        print(f"  {'-'*75}")

        for mode_label in ['distance', 'random']:
            for pair_type in ['similar', 'dissimilar']:
                vals = defaultdict(list)
                for r in all_results:
                    if r['mode'] != mode_label:
                        continue
                    is_sim = (r['cls_A'], r['cls_B']) in similar_pairs
                    if (pair_type == 'similar') != is_sim:
                        continue
                    for mn in ['cosine_similarity', 'importance_ratio', 'topk_jaccard', 'directional_interference']:
                        vals[mn].append(r['metrics'][wk][mn])

                if vals:
                    cs = np.mean(vals['cosine_similarity'])
                    ir = np.mean(vals['importance_ratio'])
                    jc = np.mean(vals['topk_jaccard'])
                    di = np.mean(vals['directional_interference'])
                    print(f"  {mode_label:<15} {pair_type:<12} {cs:>12.4f} {ir:>12.4f} {jc:>10.4f} {di:>12.6f}")

    # ===== Plots =====
    plot_interference_comparison(all_results, save_dir)
    plot_directional_interference(all_results, save_dir)

    print(f"\nResults and plots saved to {save_dir}/")


# ─────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Two-class synapse interference measurement')

    # Model
    parser.add_argument('--sigma', type=float, default=0.7)
    parser.add_argument('--target_mean', type=float, default=0.05)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Seeds
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_seeds', type=int, default=3)

    # Training
    parser.add_argument('--per_class', type=int, default=10)
    parser.add_argument('--T_on', type=int, default=1000)
    parser.add_argument('--T_off', type=int, default=2000)
    parser.add_argument('--T_skip', type=int, default=100)

    # Burn-in
    parser.add_argument('--burn_in_on', type=int, default=2000)
    parser.add_argument('--burn_in_off', type=int, default=3000)

    # Output
    parser.add_argument('--save_dir', type=str,
                        default='outputs/step3_synapse_interference')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
