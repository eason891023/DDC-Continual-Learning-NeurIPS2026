"""
Class-Incremental 10-Way MNIST with Sigma Sweep (Multi-Seed)

Conditions: σ=0.3, 0.5, 0.7, 1.0 (mode=distance) + mode=random
Each condition × N seeds → aggregate mean±std + statistical tests

Protocol:
  - 10 tasks, 1 class per task (class 0→1→...→9)
  - Accumulate spike counts into neurons_labels during training (plasticity ON)
  - Selective readout: neurons_labels - row_mean → matmul → argmax
  - Interim test after each task → forgetting curve
  - Final test after all 10 tasks

Usage:
    python -m experiments.cl_sigma_sweep --device cuda --n_seeds 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from main.Network import Brain_Like_RSNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from scipy import stats


# ─────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────

def get_per_class_subsets(dataset, per_class, seed=42):
    rng = np.random.RandomState(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    subsets = {}
    for label in range(10):
        indices = class_indices[label]
        rng.shuffle(indices)
        subsets[label] = Subset(dataset, indices[:per_class])
    return subsets


def parse_class_order(class_order_arg):
    if class_order_arg is None:
        return list(range(10))

    if isinstance(class_order_arg, (list, tuple)):
        order = [int(p) for p in class_order_arg]
        if len(order) != 10:
            raise ValueError(
                f'class order list must contain exactly 10 class ids, got {len(order)}'
            )
        if sorted(order) != list(range(10)):
            raise ValueError('class order list must be a permutation of 0,1,2,3,4,5,6,7,8,9')
        return order

    parts = [p.strip() for p in class_order_arg.split(',') if p.strip()]
    if len(parts) != 10:
        raise ValueError(
            f'--class_order must contain exactly 10 comma-separated class ids, got {len(parts)}'
        )

    try:
        order = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError('--class_order must contain integers only') from exc

    if sorted(order) != list(range(10)):
        raise ValueError('--class_order must be a permutation of 0,1,2,3,4,5,6,7,8,9')

    return order


def format_order_tag(class_order):
    return ''.join(str(c) for c in class_order)


def format_condition_name(mode, sigma):
    """Return the run name for one topology condition."""
    if mode == 'distance':
        return f"distance_sigma{sigma}"
    return str(mode)


# ─────────────────────────────────────────────────
#  Interim evaluation
# ─────────────────────────────────────────────────

READOUT_METHODS = ['selective']


def _build_readout_variants(neurons_labels):
    """Build the selective-vote readout matrix (mean-centered class columns)."""
    row_mean = neurons_labels.mean(dim=1, keepdim=True)
    selective = neurons_labels - row_mean
    return {'selective': (selective, 'dot')}


def evaluate(model, test_loader, neurons_labels, T_on, T_off, T_skip, device,
             desc="Eval", state_policy="legacy_continuous"):
    """
    Evaluate the selective-vote readout.

    ``legacy_continuous`` preserves the historical paper behavior: recurrent
    state is carried from one batch/sample to the next. ``sample_reset``
    rebuilds a neutral dynamic state before every loader batch. With
    ``batch_size=1`` this is an exact sample-independent evaluation; larger
    batches contain independent rows but require a separate equivalence audit.

    Returns: dict of {method: (accuracy, per_class_acc, confusion)}
    """
    if state_policy not in {"legacy_continuous", "sample_reset"}:
        raise ValueError(
            "state_policy must be 'legacy_continuous' or 'sample_reset', "
            f"got {state_policy!r}"
        )
    model.disable_plasticity()

    n_classes = 10
    readouts = _build_readout_variants(neurons_labels)

    # Per-method accumulators
    method_stats = {}
    for name in READOUT_METHODS:
        method_stats[name] = {
            'correct': 0, 'total': 0,
            'per_class_correct': torch.zeros(n_classes, device=device),
            'per_class_total': torch.zeros(n_classes, device=device),
            'confusion': torch.zeros((n_classes, n_classes), device=device),
        }

    orig_batch_size = int(model.RSNN_Exc_pre_spikes.shape[0]) if model.RSNN_Exc_pre_spikes is not None else 1
    sample_shape = None

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc=desc, leave=False)):
            x = x.to(device)
            y = y.to(device)
            batch_size = int(x.shape[0])

            if sample_shape is None:
                sample_shape = tuple(x.shape[1:])

            # ``sample_reset`` rebuilds all dynamic neuron/synapse buffers
            # before every batch, even when its size is unchanged. This removes
            # both cross-class and within-class state carryover.
            current_state_bs = int(model.RSNN_Exc_pre_spikes.shape[0])
            if state_policy == "sample_reset" or current_state_bs != batch_size:
                model.build(x)

            # Collect spike vector for this sample (one forward pass)
            spike_accum = torch.zeros((batch_size, model.exc_neurons), device=device)

            for t in range(T_on):
                exc_spikes, _ = model(x)
                if t >= T_skip:
                    spike_accum += exc_spikes
                model.step()

            zero_input = torch.zeros_like(x)
            for t in range(T_off):
                exc_spikes, _ = model(zero_input)
                spike_accum += exc_spikes
                model.step()

            # Score with each readout method
            for name in READOUT_METHODS:
                mat, _ = readouts[name]
                scores = torch.matmul(spike_accum, mat)
                preds = scores.argmax(dim=1)
                st = method_stats[name]
                for true_label, pred in zip(y.tolist(), preds.tolist()):
                    st['confusion'][true_label, pred] += 1
                    st['per_class_total'][true_label] += 1
                    if pred == true_label:
                        st['correct'] += 1
                        st['per_class_correct'][true_label] += 1
                    st['total'] += 1

    model.enable_plasticity()

    # Restore the original batch-size state so subsequent training/evaluation
    # calls retain the exact legacy behavior when the caller uses batch_size=1.
    if sample_shape is not None and int(model.RSNN_Exc_pre_spikes.shape[0]) != orig_batch_size:
        restore_x = torch.zeros((orig_batch_size, *sample_shape), device=device)
        model.build(restore_x)

    results = {}
    for name in READOUT_METHODS:
        st = method_stats[name]
        acc = st['correct'] / st['total'] if st['total'] > 0 else 0
        per_class_acc = {}
        for c in range(n_classes):
            if st['per_class_total'][c] > 0:
                per_class_acc[c] = (st['per_class_correct'][c] / st['per_class_total'][c]).item()
        results[name] = (acc, per_class_acc, st['confusion'].cpu().numpy())

    return results


def save_task_snapshot(run_dir, task_idx, cls, model, neurons_labels, snapshot_weights):
    """
    Save lightweight per-task artifacts so we can later compute benchmark-level
    directional interference without rerunning training.
    """
    snap_dir = os.path.join(run_dir, 'task_snapshots', f'task_{task_idx:02d}_class_{cls}')
    os.makedirs(snap_dir, exist_ok=True)

    torch.save(neurons_labels.cpu(), os.path.join(snap_dir, 'neurons_labels.pt'))

    if snapshot_weights in ('ee', 'both'):
        torch.save(
            model.RSNN_EE_synapase.RSNN_EE_weights.detach().cpu(),
            os.path.join(snap_dir, 'RSNN_EE_weights.pt'),
        )

    if snapshot_weights in ('input', 'both'):
        torch.save(
            model.inputs2RSNN_EE_synapase.inputs_EE_weights.detach().cpu(),
            os.path.join(snap_dir, 'inputs2RSNN_EE_weights.pt'),
        )

    meta = {
        'task_idx': int(task_idx),
        'class_learned': int(cls),
        'snapshot_weights': snapshot_weights,
    }
    with open(os.path.join(snap_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────
#  Single run (one condition × one seed)
# ─────────────────────────────────────────────────

def run_single(args, mode, sigma, seed, mnist_train, mnist_test, class_order=None, order_tag=None):
    device = args.device
    set_seed(seed)

    if class_order is None:
        class_order = parse_class_order(getattr(args, 'class_order', None))
    if order_tag is None:
        order_tag = getattr(args, 'order_tag', format_order_tag(class_order))

    cond_name = format_condition_name(mode, sigma)
    condition_suffix = getattr(args, 'condition_suffix', None)
    if condition_suffix:
        cond_name = f"{cond_name}__{condition_suffix}"
    print(f"\n{'#'*60}")
    print(f"  Condition: {cond_name} | Seed: {seed}")
    print(f"  Class order: {class_order}")
    print(f"{'#'*60}")

    # Data subsets (seed-dependent for training, fixed for testing)
    train_subsets = get_per_class_subsets(mnist_train, per_class=args.per_class, seed=seed)

    # Fixed interim subsets (a loader over the classes seen so far is built
    # after each task only when interim evaluation is requested).
    interim_subsets = None
    if not args.skip_interim:
        interim_subsets = get_per_class_subsets(mnist_test, per_class=args.interim_test_per_class, seed=42)

    # Final test is restricted to the active stream classes.
    final_subsets = get_per_class_subsets(mnist_test, per_class=args.test_per_class, seed=42)
    final_indices = []
    for cls in class_order:
        final_indices.extend(final_subsets[cls].indices)
    final_loader = DataLoader(
        Subset(mnist_test, final_indices),
        batch_size=getattr(args, 'eval_batch_size', 1),
        shuffle=False,
    )

    # Build model
    set_seed(seed)
    model = Brain_Like_RSNN(
        inputs_size=torch.Size([1, 1, 28, 28]),
        exc_neurons=4096, inh_neurons=1024,
        sigma=sigma, dt=0.1, mode=mode,
        target_mean=args.target_mean, seed=seed, device=device
    ).to(device)

    plasticity_variant = getattr(args, 'plasticity_variant', 'full')
    if plasticity_variant == 'triplet_only_exc':
        # Remove heterosynaptic and transmitter-induced terms from both
        # excitatory plastic pathways while retaining their triplet STDP and
        # the original inhibitory homeostatic plasticity.
        for synapse in (
            model.inputs2RSNN_EE_synapase,
            model.RSNN_EE_synapase,
        ):
            synapse.ltp.beta = 0.0
            synapse.ltp.delta = 0.0
    elif plasticity_variant != 'full':
        raise ValueError(
            "plasticity_variant must be 'full' or 'triplet_only_exc', "
            f"got {plasticity_variant!r}"
        )

    sample_img, _ = train_subsets[class_order[0]][0]
    model.build(sample_img.unsqueeze(0).to(device))

    # Saving full dense checkpoints costs roughly 1 GB per run. Lightweight
    # screens that only need aggregate accuracy can opt out without changing
    # the historical default used by existing experiments.
    save_checkpoints = bool(getattr(args, 'save_checkpoints', True))

    # Save init checkpoint
    run_dir = os.path.join(args.save_dir, 'runs', f'{cond_name}_order{order_tag}_seed{seed}')
    os.makedirs(run_dir, exist_ok=True)
    if save_checkpoints:
        model.save_checkpoint(os.path.join(run_dir, 'checkpoints', 'init'), tag='init')

    # Burn-in
    all_train_indices = []
    for cls in class_order:
        all_train_indices.extend(train_subsets[cls].indices)
    burn_loader = DataLoader(Subset(mnist_train, all_train_indices),
                              batch_size=1, shuffle=False)
    model.burn_in(burn_loader, T_on=args.burn_in_on, T_off=args.burn_in_off)

    # Class-incremental training
    neurons_labels = torch.zeros((model.exc_neurons, 10), device=device)
    T_on = args.T_on
    T_off = args.T_off
    T_skip = args.T_skip

    # Results storage
    interim_accs = []  # accuracy after each task
    interim_per_class = []  # per-class accuracy after each task

    for task_idx, cls in enumerate(class_order):
        loader = DataLoader(train_subsets[cls], batch_size=1, shuffle=False)
        print(f"\n  Task {task_idx}: Class {cls} ({len(train_subsets[cls])} samples)")

        # Training
        model.enable_plasticity()
        with torch.no_grad():
            for img, label in tqdm(loader, desc=f"  Train cls {cls}", leave=False):
                x = img.to(device)

                for t in range(T_on):
                    exc_spikes, _ = model(x)
                    model.step()
                    if t >= T_skip:
                        neurons_labels[:, cls] += exc_spikes[0]

                zero_input = torch.zeros_like(x)
                for t in range(T_off):
                    exc_spikes, _ = model(zero_input)
                    model.step()
                    neurons_labels[:, cls] += exc_spikes[0]

        # Interim evaluation (optional, skipped by default for speed)
        if not args.skip_interim:
            seen_classes = class_order[:task_idx + 1]
            interim_indices = []
            for seen_cls in seen_classes:
                interim_indices.extend(interim_subsets[seen_cls].indices)
            interim_loader = DataLoader(
                Subset(mnist_test, interim_indices),
                batch_size=getattr(args, 'eval_batch_size', 1),
                shuffle=False,
            )
            eval_results = evaluate(
                model, interim_loader, neurons_labels,
                T_on, T_off, T_skip, device,
                desc=f"  Eval after task {task_idx}",
                state_policy=getattr(args, 'eval_state_policy', 'legacy_continuous'),
            )
            # Use 'selective' (original method) for interim tracking
            acc, per_cls_acc, _ = eval_results['selective']
            interim_accs.append(acc)
            interim_per_class.append(per_cls_acc)
            print(f"  After task {task_idx} (learned 0-{cls}): "
                  f"Overall Acc = {acc*100:.2f}%")

        if args.save_task_snapshots:
            save_task_snapshot(
                run_dir=run_dir,
                task_idx=task_idx,
                cls=cls,
                model=model,
                neurons_labels=neurons_labels,
                snapshot_weights=args.task_snapshot_weights,
            )

    # Save trained checkpoint
    if save_checkpoints:
        model.save_checkpoint(os.path.join(run_dir, 'checkpoints', 'trained'), tag='trained')
    torch.save(neurons_labels.cpu(), os.path.join(run_dir, 'neurons_labels.pt'))

    # Final evaluation with full test set + confusion matrix (all readout methods)
    eval_results = evaluate(
        model, final_loader, neurons_labels,
        T_on, T_off, T_skip, device,
        desc="  Final eval",
        state_policy=getattr(args, 'eval_state_policy', 'legacy_continuous'),
    )

    print(f"\n  Final Accuracy by readout method:")
    for method in READOUT_METHODS:
        acc, per_cls, _ = eval_results[method]
        print(f"    [{method}] Overall: {acc*100:.2f}%")
        for c in range(10):
            if c in per_cls:
                print(f"      Class {c}: {per_cls[c]*100:.2f}%")

    del model
    if device != 'cpu':
        torch.cuda.empty_cache()

    # Pack results for all methods
    result = {
        'condition': cond_name,
        'mode': mode,
        'evaluation_state_policy': getattr(
            args, 'eval_state_policy', 'legacy_continuous'
        ),
        'evaluation_batch_size': int(getattr(args, 'eval_batch_size', 1)),
        'full_checkpoints_saved': save_checkpoints,
        'plasticity_variant': plasticity_variant,
        'sigma': sigma,
        'seed': seed,
        'class_order': class_order,
        'order_tag': order_tag,
        'interim_accs': interim_accs,
    }
    for method in READOUT_METHODS:
        acc, per_cls, confusion = eval_results[method]
        result[f'final_acc_{method}'] = acc
        result[f'final_per_class_{method}'] = {str(k): v for k, v in per_cls.items()}
        result[f'confusion_{method}'] = confusion.tolist()

    # Keep backward-compatible keys (point to 'selective' = original method)
    result['final_acc'] = result['final_acc_selective']
    result['final_per_class'] = result['final_per_class_selective']
    result['confusion'] = result['confusion_selective']

    return result


# ─────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────

def plot_forgetting_curves(all_results, save_dir, n_seeds, n_tasks=10):
    """Forgetting curves: accuracy after each task, per condition."""
    import matplotlib.pyplot as plt

    conditions = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(conditions)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, color in zip(conditions, colors):
        seed_curves = all_results[cond]['interim_accs_all']
        mean_curve = np.mean(seed_curves, axis=0) * 100
        std_curve = np.std(seed_curves, axis=0) * 100
        x = np.arange(1, n_tasks + 1)

        ax.plot(x, mean_curve, 'o-', color=color, label=cond, linewidth=2, markersize=5)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                         color=color, alpha=0.15)

    ax.set_xlabel('Number of Classes Learned', fontsize=11)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=11)
    ax.set_title(f'{n_tasks}-Way Class-IL Forgetting Curves (N={n_seeds} seeds)', fontsize=13)
    ax.set_xticks(range(1, n_tasks + 1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'forgetting_curves.png'), dpi=200)
    plt.savefig(os.path.join(save_dir, 'forgetting_curves.pdf'))
    print("Plot saved: forgetting_curves.png")
    plt.close()


def plot_final_accuracy_bar(all_results, save_dir, n_seeds, n_tasks=10):
    """Bar chart of final accuracy per condition with error bars."""
    import matplotlib.pyplot as plt

    conditions = list(all_results.keys())
    means = [all_results[c]['final_acc_mean'] * 100 for c in conditions]
    stds = [all_results[c]['final_acc_std'] * 100 for c in conditions]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(conditions)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.5, capsize=5, error_kw={'linewidth': 1.2})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                f'{m:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel('Final Accuracy (%)', fontsize=11)
    ax.set_title(f'{n_tasks}-Way Class-IL Final Accuracy (N={n_seeds} seeds)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_accuracy.png'), dpi=200)
    plt.savefig(os.path.join(save_dir, 'final_accuracy.pdf'))
    print("Plot saved: final_accuracy.png")
    plt.close()


def plot_dose_response(all_results, save_dir, n_seeds):
    """Dose-response: sigma vs final accuracy (distance conditions only)."""
    import matplotlib.pyplot as plt

    dist_conds = [(c, r) for c, r in all_results.items() if r['mode'] == 'distance']
    if not dist_conds:
        return

    dist_conds.sort(key=lambda x: x[1]['sigma'])
    sigmas = [r['sigma'] for _, r in dist_conds]
    means = [r['final_acc_mean'] * 100 for _, r in dist_conds]
    stds = [r['final_acc_std'] * 100 for _, r in dist_conds]

    # Add random as reference
    rand_mean = all_results.get('random', {}).get('final_acc_mean', 0) * 100
    rand_std = all_results.get('random', {}).get('final_acc_std', 0) * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(sigmas, means, yerr=stds, fmt='o-', color='#2196F3',
                linewidth=2, markersize=8, capsize=5, label='Distance')

    # Individual seed dots
    for _, r in dist_conds:
        vals = [v * 100 for v in r['final_acc_values']]
        ax.scatter([r['sigma']] * len(vals), vals, c='#2196F3', s=15,
                   alpha=0.4, zorder=3)

    if rand_mean > 0:
        ax.axhline(y=rand_mean, color='#FF5722', linestyle='--', linewidth=2,
                   label=f'Random ({rand_mean:.1f}±{rand_std:.1f}%)')
        ax.axhspan(rand_mean - rand_std, rand_mean + rand_std,
                    color='#FF5722', alpha=0.1)

    ax.set_xlabel('σ (connectivity width)', fontsize=11)
    ax.set_ylabel('Final Accuracy (%)', fontsize=11)
    ax.set_title(f'Dose-Response: σ vs CL Performance (N={n_seeds} seeds)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dose_response.png'), dpi=200)
    plt.savefig(os.path.join(save_dir, 'dose_response.pdf'))
    print("Plot saved: dose_response.png")
    plt.close()


def plot_per_class_heatmap(all_results, save_dir, class_order=None):
    """Heatmap: per-class accuracy for each condition (mean across seeds)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    conditions = list(all_results.keys())
    if class_order is None:
        class_order = list(range(10))
    matrix = np.zeros((len(conditions), len(class_order)))

    for i, cond in enumerate(conditions):
        for j, c in enumerate(class_order):
            vals = all_results[cond]['per_class_all'].get(str(c), [])
            matrix[i, j] = np.mean(vals) * 100 if vals else 0

    fig, ax = plt.subplots(figsize=(12, max(3, len(conditions) * 0.8)))
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=[str(i) for i in class_order],
                yticklabels=conditions, ax=ax, vmin=0, vmax=100)
    ax.set_xlabel('Class')
    ax.set_ylabel('Condition')
    ax.set_title('Per-Class Accuracy (%) — Mean across seeds')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_heatmap.png'), dpi=200)
    plt.savefig(os.path.join(save_dir, 'per_class_heatmap.pdf'))
    print("Plot saved: per_class_heatmap.png")
    plt.close()


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────

def run_experiment(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    seeds = list(range(args.seed, args.seed + args.n_seeds))

    # Define conditions
    conditions = []
    for sigma in args.sigmas:
        conditions.append(('distance', sigma, f'dist_σ={sigma}'))
    if not args.no_random:
        conditions.append(('random', 1.0, 'random'))

    # Load dataset once
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'fmnist':
        mnist_train = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
        mnist_test = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)
    else:
        mnist_train = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    # Collect all results (condition-major order)
    all_raw = {cond_name: [] for _, _, cond_name in conditions}
    jobs = [(*condition, seed) for condition in conditions for seed in seeds]

    for mode, sigma, cond_name, seed in jobs:
        # Write result.json next to the run's other artifacts (run_single uses the same name).
        result_path = os.path.join(
            save_dir, 'runs',
            f'{format_condition_name(mode, sigma)}_order{args.order_tag}_seed{seed}',
            'result.json',
        )
        if getattr(args, 'resume', False) and os.path.exists(result_path):
            print(f"Resuming from completed run: {result_path}")
            with open(result_path) as f:
                result = json.load(f)
        else:
            result = run_single(args, mode, sigma, seed, mnist_train, mnist_test)

            # Save per-run
            run_dir = os.path.dirname(result_path)
            os.makedirs(run_dir, exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
        all_raw[cond_name].append(result)

    # ===== Aggregate (per readout method) =====
    aggregated = {}          # {method: {cond_name: agg_dict}}
    for method in READOUT_METHODS:
        aggregated[method] = {}
        for cond_name, runs in all_raw.items():
            final_accs = [r[f'final_acc_{method}'] for r in runs]
            interim_accs_all = [r['interim_accs'] for r in runs]

            per_class_all = defaultdict(list)
            for r in runs:
                for c, acc in r[f'final_per_class_{method}'].items():
                    per_class_all[c].append(acc)

            aggregated[method][cond_name] = {
                'mode': runs[0]['mode'],
                'sigma': runs[0]['sigma'],
                'final_acc_mean': float(np.mean(final_accs)),
                'final_acc_std': float(np.std(final_accs)),
                'final_acc_values': final_accs,
                'interim_accs_all': interim_accs_all,
                'per_class_all': dict(per_class_all),
            }

    # ===== Statistical tests (per method) =====
    all_stat_tests = {}
    for method in READOUT_METHODS:
        stat_tests = {}
        agg_m = aggregated[method]

        rand_accs = agg_m.get('random', {}).get('final_acc_values', [])
        for cond_name, agg in agg_m.items():
            if cond_name == 'random':
                continue
            dist_accs = agg['final_acc_values']
            if len(dist_accs) >= 3 and len(rand_accs) >= 3:
                t_stat, p_val = stats.ttest_ind(dist_accs, rand_accs)
                u_stat, u_p = stats.mannwhitneyu(dist_accs, rand_accs, alternative='two-sided')
            else:
                t_stat, p_val = np.nan, np.nan
                u_stat, u_p = np.nan, np.nan

            stat_tests[f'{cond_name}_vs_random'] = {
                't_stat': float(t_stat),
                'p_value': float(p_val),
                'mann_whitney_U': float(u_stat) if not np.isnan(u_stat) else None,
                'mann_whitney_p': float(u_p) if not np.isnan(u_p) else None,
                'dist_values': dist_accs,
                'rand_values': rand_accs,
            }

        # Spearman correlation: sigma vs accuracy (distance only)
        dist_conds = [(agg['sigma'], agg['final_acc_mean'])
                       for name, agg in agg_m.items() if agg['mode'] == 'distance']
        if len(dist_conds) >= 3:
            dist_conds.sort()
            sigmas_list = [s for s, _ in dist_conds]
            accs_list = [a for _, a in dist_conds]
            rho, rho_p = stats.spearmanr(sigmas_list, accs_list)
            stat_tests['sigma_acc_correlation'] = {
                'spearman_rho': float(rho),
                'p_value': float(rho_p),
                'sigmas': sigmas_list,
                'accs': accs_list,
            }
        all_stat_tests[method] = stat_tests

    # ===== Save aggregate =====
    agg_json = {}
    for method in READOUT_METHODS:
        agg_json[method] = {}
        for cond_name, agg in aggregated[method].items():
            agg_json[method][cond_name] = {
                'mode': agg['mode'],
                'sigma': agg['sigma'],
                'final_acc_mean': agg['final_acc_mean'],
                'final_acc_std': agg['final_acc_std'],
                'final_acc_values': agg['final_acc_values'],
                'interim_accs_mean': np.mean(agg['interim_accs_all'], axis=0).tolist() if agg['interim_accs_all'] and len(agg['interim_accs_all'][0]) > 0 else [],
                'interim_accs_std': np.std(agg['interim_accs_all'], axis=0).tolist() if agg['interim_accs_all'] and len(agg['interim_accs_all'][0]) > 0 else [],
                'per_class_mean': {c: float(np.mean(v)) for c, v in agg['per_class_all'].items()},
                'per_class_std': {c: float(np.std(v)) for c, v in agg['per_class_all'].items()},
            }
        agg_json[method]['stat_tests'] = all_stat_tests[method]

    with open(os.path.join(save_dir, 'aggregate.json'), 'w') as f:
        json.dump(agg_json, f, indent=2)
    print(f"\nAggregate saved to {save_dir}/aggregate.json")

    # ===== Plots (use 'selective' = original method for existing plots) =====
    agg_default = aggregated['selective']
    has_interim = any(len(agg['interim_accs_all'][0]) > 0 for agg in agg_default.values()
                       if agg['interim_accs_all'])
    if has_interim:
        plot_forgetting_curves(agg_default, save_dir, args.n_seeds)
    plot_final_accuracy_bar(agg_default, save_dir, args.n_seeds)
    plot_dose_response(agg_default, save_dir, args.n_seeds)
    plot_per_class_heatmap(agg_default, save_dir, args.class_order)

    # ===== Print summary (all methods) =====
    for method in READOUT_METHODS:
        agg_m = aggregated[method]
        print(f"\n{'='*80}")
        print(f"  READOUT: {method.upper()}  (N={args.n_seeds} seeds)")
        print(f"{'='*80}")
        print(f"{'Condition':<20} {'Final Acc':>15}")
        print(f"{'-'*40}")
        for cond_name in agg_m:
            agg = agg_m[cond_name]
            m = agg['final_acc_mean'] * 100
            s = agg['final_acc_std'] * 100
            print(f"{cond_name:<20} {m:>6.2f}±{s:<.2f}%")
        print(f"{'-'*40}")

        st = all_stat_tests[method]
        print(f"  Statistical tests (vs random):")
        for test_name, test_val in st.items():
            if test_name == 'sigma_acc_correlation':
                print(f"    σ-accuracy correlation: ρ={test_val['spearman_rho']:.3f}, "
                      f"p={test_val['p_value']:.4f}")
            else:
                p = test_val['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                print(f"    {test_name}: t={test_val['t_stat']:.3f}, p={p:.4f} {sig}")

    print(f"\nAll results saved to {save_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description='CL Sigma Sweep: Class-Incremental MNIST')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.3, 0.5, 0.7, 1.0],
                        help='Sigma values to test (distance mode)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist'],
                        help='Dataset to use: mnist or fmnist (FashionMNIST)')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Directory where torchvision datasets are downloaded')
    parser.add_argument('--class_order', type=str, default=None,
                        help='Comma-separated class order permutation, e.g. 0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--no_random', action='store_true',
                        help='Skip the random-connectivity control condition')
    parser.add_argument('--target_mean', type=float, default=0.05)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Seeds
    parser.add_argument('--seed', type=int, default=47, help='Starting seed')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed per-run result.json files in save_dir')

    # Training
    parser.add_argument('--per_class', type=int, default=10,
                        help='Training samples per class')
    parser.add_argument('--T_on', type=int, default=1000)
    parser.add_argument('--T_off', type=int, default=2000)
    parser.add_argument('--T_skip', type=int, default=100)

    # Testing
    parser.add_argument('--test_per_class', type=int, default=10)
    parser.add_argument('--interim_test_per_class', type=int, default=10,
                        help='Test samples per class for interim evaluation (small for speed)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Batch size for interim/final evaluation only. '
                             'Training remains batch_size=1 to preserve the original protocol.')
    parser.add_argument(
        '--eval_state_policy',
        choices=['legacy_continuous', 'sample_reset'],
        default='legacy_continuous',
        help=(
            'Dynamic-state convention during evaluation. legacy_continuous '
            'matches the paper; sample_reset rebuilds neutral state before each batch.'
        ),
    )
    parser.add_argument('--skip_interim', action='store_true', default=True,
                        help='Skip interim evaluation after each task (faster)')
    parser.add_argument('--no_skip_interim', dest='skip_interim', action='store_false',
                        help='Enable interim evaluation after each task')
    parser.add_argument('--save_task_snapshots', action='store_true',
                        help='Save per-task neurons_labels and selected weights for later interference analysis')
    parser.add_argument('--task_snapshot_weights', type=str, default='both',
                        choices=['none', 'ee', 'input', 'both'],
                        help='Which weights to store when --save_task_snapshots is enabled')
    parser.add_argument(
        '--skip_checkpoints',
        dest='save_checkpoints',
        action='store_false',
        default=True,
        help='Do not save dense init/trained checkpoints (for lightweight screens).',
    )

    # Burn-in
    parser.add_argument('--burn_in_on', type=int, default=2000)
    parser.add_argument('--burn_in_off', type=int, default=3000)

    # Output
    parser.add_argument('--save_dir', type=str,
                        default='outputs/benchmark_mnist')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Auto-suffix save_dir with dataset name if still at default
    if args.save_dir == 'outputs/benchmark_mnist' and args.dataset != 'mnist':
        args.save_dir = f'outputs/benchmark_{args.dataset}'
    args.class_order = parse_class_order(args.class_order)
    args.order_tag = format_order_tag(args.class_order)
    run_experiment(args)
