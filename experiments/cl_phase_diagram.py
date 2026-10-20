"""
Representational phase diagram experiment.

Collects per-sample firing rates + mean membrane potentials:
  1. During class-incremental training (plasticity ON), one rate vector per training
     sample as it is learned.
  2. After all training, on a fixed test set (plasticity OFF), one rate vector per
     test sample.

Training and test images are fixed across all runs (DATA_SEED). Only the model seed
varies across runs.

Output per run (under --save_dir/runs/<cond>_seed<s>/):
    training_trajectory.pt   H_rate [N_train, 5120], H_V [...], preact [N_train, 10],
                             labels [N_train], task_map [N_train], step_in_task [N_train]
    test_final.pt            H_rate [N_test, 5120], H_V [...], preact [N_test, 10],
                             labels [N_test]
    neurons_labels_snapshots.pt  [N_tasks+1, N_exc, 10]  (pre-training + post-task k)
    neurons_labels_final.pt  same as snapshots[-1], kept for convenience
    result.json              final accuracy under all readout variants
    checkpoints/{init,trained}  full model checkpoints
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from main.Network import Brain_Like_RSNN
from experiments.cl_sigma_sweep import (
    set_seed,
    parse_class_order,
    format_order_tag,
    READOUT_METHODS,
    _build_readout_variants,
)


DATA_SEED = 777  # Fixed: selects training/test image indices so all runs use identical data.


# ─────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────

def build_fixed_subsets(dataset, per_class, rng_seed):
    """Pick `per_class` indices per class with a fixed RNG seed."""
    rng = np.random.RandomState(rng_seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    subsets = {}
    picked_indices = {}
    for label in range(10):
        indices = list(class_indices[label])
        rng.shuffle(indices)
        picked = indices[:per_class]
        subsets[label] = Subset(dataset, picked)
        picked_indices[label] = picked
    return subsets, picked_indices


def flatten_picked_indices(picked_indices, class_order):
    flat = []
    for cls in class_order:
        flat.extend(picked_indices[cls])
    return flat


def indices_hash(indices):
    arr = np.array(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def build_eval_subset(test_picked, class_order, upto_task_idx):
    """Return fixed test indices for classes learned up to `upto_task_idx`."""
    seen_classes = list(class_order[:upto_task_idx + 1])
    indices = []
    for cls in seen_classes:
        indices.extend(test_picked[cls])
    return seen_classes, indices


def init_eval_stats(n_classes):
    stats = {}
    for name in READOUT_METHODS:
        stats[name] = {
            'correct': 0,
            'total': 0,
            'per_class_correct': torch.zeros(n_classes),
            'per_class_total': torch.zeros(n_classes),
            'confusion': torch.zeros((n_classes, n_classes)),
        }
    return stats


def finalize_eval_stats(stats):
    results = {}
    for name in READOUT_METHODS:
        st = stats[name]
        acc = st['correct'] / st['total'] if st['total'] > 0 else 0.0
        per_class_acc = {}
        for c in range(st['per_class_total'].shape[0]):
            if st['per_class_total'][c] > 0:
                per_class_acc[c] = float((st['per_class_correct'][c] / st['per_class_total'][c]).item())
        results[name] = (acc, per_class_acc, st['confusion'].numpy())
    return results


def probe_dataset(model, loader, dataset_indices, neurons_labels, T_on, T_off, T_skip,
                  device, desc="Probe", store_voltage=False):
    """
    Collect fixed test representations with plasticity OFF and, in the same pass,
    compute readout accuracies for all methods.
    """
    n_classes = 10
    N_exc = model.exc_neurons
    N_inh = model.inh_neurons
    H_dim = N_exc + N_inh
    n_samples = len(dataset_indices)
    T_rate = (T_on - T_skip) + T_off
    T_V = T_rate

    payload = {
        'H_rate': torch.zeros((n_samples, H_dim), dtype=torch.float32),
        'preact': torch.zeros((n_samples, n_classes), dtype=torch.float32),
        'labels': torch.zeros(n_samples, dtype=torch.long),
        'dataset_indices': torch.tensor(dataset_indices, dtype=torch.long),
        'N_exc': N_exc,
        'N_inh': N_inh,
    }
    if store_voltage:
        payload['H_V'] = torch.zeros((n_samples, H_dim), dtype=torch.float32)

    prev_plasticity = bool(getattr(model, 'plasticity_enabled', True))
    model.disable_plasticity()
    readouts = _build_readout_variants(neurons_labels)
    stats = init_eval_stats(n_classes)

    offset = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device)
            y = y.to(device)
            batch_size = int(x.shape[0])
            if batch_size != 1:
                raise ValueError(
                    f"{desc}: eval/probe batch_size must be 1, got {batch_size}. "
                    "This experiment stores and probes exact network state."
                )

            current_state_bs = int(model.RSNN_Exc_pre_spikes.shape[0])
            if current_state_bs != batch_size:
                raise RuntimeError(
                    f"{desc}: model state batch dimension is {current_state_bs}, "
                    f"but input batch dimension is {batch_size}. Refusing to rebuild, "
                    "because build() would reset the saved network state."
                )

            spike_exc = torch.zeros((batch_size, N_exc), device=device)
            spike_inh = torch.zeros((batch_size, N_inh), device=device)
            if store_voltage:
                v_exc = torch.zeros((batch_size, N_exc), device=device)
                v_inh = torch.zeros((batch_size, N_inh), device=device)

            for t in range(T_on):
                exc_spikes, inh_spikes = model(x)
                model.step()
                if t >= T_skip:
                    spike_exc += exc_spikes
                    spike_inh += inh_spikes
                    if store_voltage:
                        v_exc += model.RSNN_Exc_neurons.v
                        v_inh += model.RSNN_Inh_neurons.v

            zero_input = torch.zeros_like(x)
            for _ in range(T_off):
                exc_spikes, inh_spikes = model(zero_input)
                model.step()
                spike_exc += exc_spikes
                spike_inh += inh_spikes
                if store_voltage:
                    v_exc += model.RSNN_Exc_neurons.v
                    v_inh += model.RSNN_Inh_neurons.v

            rate_exc = (spike_exc / T_rate).detach().cpu()
            rate_inh = (spike_inh / T_rate).detach().cpu()
            payload['H_rate'][offset:offset + batch_size, :N_exc] = rate_exc
            payload['H_rate'][offset:offset + batch_size, N_exc:] = rate_inh
            payload['labels'][offset:offset + batch_size] = y.detach().cpu()

            selective = readouts['selective'][0]
            payload['preact'][offset:offset + batch_size] = torch.matmul(
                rate_exc.to(device), selective
            ).detach().cpu()

            if store_voltage:
                payload['H_V'][offset:offset + batch_size, :N_exc] = (v_exc / T_V).detach().cpu()
                payload['H_V'][offset:offset + batch_size, N_exc:] = (v_inh / T_V).detach().cpu()

            for name in READOUT_METHODS:
                mat, _ = readouts[name]
                scores = torch.matmul(spike_exc, mat)
                preds = scores.argmax(dim=1)
                st = stats[name]
                for true_label, pred in zip(y.tolist(), preds.tolist()):
                    st['confusion'][true_label, pred] += 1
                    st['per_class_total'][true_label] += 1
                    if pred == true_label:
                        st['correct'] += 1
                        st['per_class_correct'][true_label] += 1
                    st['total'] += 1

            offset += batch_size

    if prev_plasticity:
        model.enable_plasticity()
    else:
        model.disable_plasticity()

    return payload, finalize_eval_stats(stats)


def save_stage_probe(run_dir, task_idx, cls, seen_classes, probe_payload, eval_results):
    stage_dir = os.path.join(run_dir, 'stage_representations')
    os.makedirs(stage_dir, exist_ok=True)
    stage_path = os.path.join(stage_dir, f'task_{task_idx:02d}_class_{cls}.pt')
    torch.save({
        **probe_payload,
        'task_idx': int(task_idx),
        'class_learned': int(cls),
        'seen_classes': torch.tensor(seen_classes, dtype=torch.long),
        'eval': {
            name: {
                'acc': float(eval_results[name][0]),
                'per_class': {int(k): float(v) for k, v in eval_results[name][1].items()},
            }
            for name in READOUT_METHODS
        },
    }, stage_path)
    return stage_path


TRACKED_MODULE_NAMES = [
    'inputs_Exc_neurons',
    'inputs_Inh_neurons',
    'RSNN_Exc_neurons',
    'RSNN_Inh_neurons',
    'inputs_EI_synapase',
    'inputs_IE_synapase',
    'inputs2RSNN_EE_synapase',
    'RSNN_EE_synapase',
    'RSNN_EI_synapase',
    'RSNN_IE_synapase',
    'RSNN_II_synapase',
]


def _state_value_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_state_value_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_state_value_to_cpu(v) for v in value)
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return None


def _state_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, list):
        return [_state_value_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_state_value_to_device(v, device) for v in value)
    return value


def _capture_runtime_attrs(obj):
    state = {}
    for name, value in vars(obj).items():
        if name.startswith('_'):
            continue
        stored = _state_value_to_cpu(value)
        if stored is not None:
            state[name] = stored
    return state


def _restore_runtime_attrs(obj, state, device):
    for name, value in state.items():
        if name == 'device':
            setattr(obj, name, device)
            continue
        if isinstance(value, dict):
            if hasattr(obj, name):
                _restore_runtime_attrs(getattr(obj, name), value, device)
            continue
        setattr(obj, name, _state_value_to_device(value, device))


def capture_complete_model_state(model, neurons_labels, task_idx=None, class_learned=None):
    """
    Save weights plus all live tensor/scalar runtime state that is advanced by
    forward()/step(). This is intentionally richer than Network.save_checkpoint().
    """
    modules = {}
    for module_name in TRACKED_MODULE_NAMES:
        module = getattr(model, module_name)
        module_state = _capture_runtime_attrs(module)
        for child_name in ['stp', 'ltp', 'inhibitory_stdp']:
            if hasattr(module, child_name):
                module_state[child_name] = _capture_runtime_attrs(getattr(module, child_name))
        modules[module_name] = module_state

    return {
        'format': 'complete_cl_stage_state_v1',
        'task_idx': None if task_idx is None else int(task_idx),
        'class_learned': None if class_learned is None else int(class_learned),
        'plasticity_enabled': bool(getattr(model, 'plasticity_enabled', True)),
        'model_attrs': _capture_runtime_attrs(model),
        'modules': modules,
        'neurons_labels': neurons_labels.detach().cpu().clone(),
        'batch_size': int(model.RSNN_Exc_pre_spikes.shape[0]),
        'N_exc': int(model.exc_neurons),
        'N_inh': int(model.inh_neurons),
    }


def apply_complete_model_state(model, state, device):
    if state.get('format') != 'complete_cl_stage_state_v1':
        raise ValueError(f"Unsupported complete state format: {state.get('format')}")

    _restore_runtime_attrs(model, state['model_attrs'], device)
    for module_name, module_state in state['modules'].items():
        module = getattr(model, module_name)
        for child_name in ['stp', 'ltp', 'inhibitory_stdp']:
            child_state = module_state.pop(child_name, None)
            if child_state is not None:
                _restore_runtime_attrs(getattr(module, child_name), child_state, device)
        _restore_runtime_attrs(module, module_state, device)

    if bool(state['plasticity_enabled']):
        model.enable_plasticity()
    else:
        model.disable_plasticity()


def save_complete_model_state(path, model, neurons_labels, task_idx=None, class_learned=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        capture_complete_model_state(model, neurons_labels, task_idx, class_learned),
        path,
    )
    return path


def build_fresh_model(args, mode, sigma, seed, device):
    set_seed(seed)
    return Brain_Like_RSNN(
        inputs_size=torch.Size([1, 1, 28, 28]),
        exc_neurons=4096, inh_neurons=1024,
        sigma=sigma, dt=0.1, mode=mode,
        input_mode=getattr(args, 'input_mode', 'spatial'),
        target_mean=args.target_mean, seed=seed, device=device,
    ).to(device)


def build_bwt_summary(class_order, stage_eval_history, bwt_matrices):
    first_learn_stage = {cls: idx for idx, cls in enumerate(class_order)}
    bwt_summary = {
        'class_order': [int(c) for c in class_order],
        'first_learn_stage': {str(k): int(v) for k, v in first_learn_stage.items()},
        'stage_eval_history': stage_eval_history,
    }
    for method in READOUT_METHODS:
        mat = bwt_matrices[method]
        per_class_bwt = {}
        for cls in class_order:
            start_stage = first_learn_stage[int(cls)]
            start_acc = mat[start_stage, int(cls)]
            final_acc = mat[len(class_order) - 1, int(cls)]
            if np.isfinite(start_acc) and np.isfinite(final_acc):
                per_class_bwt[str(int(cls))] = float(final_acc - start_acc)
            else:
                per_class_bwt[str(int(cls))] = float('nan')
        bwt_summary[f'acc_matrix_{method}'] = mat.tolist()
        bwt_summary[f'per_class_bwt_{method}'] = per_class_bwt
        valid_vals = [
            v for cls, v in per_class_bwt.items()
            if int(cls) != int(class_order[-1]) and np.isfinite(v)
        ]
        bwt_summary[f'mean_bwt_{method}'] = (
            float(np.mean(valid_vals)) if valid_vals else float('nan')
        )
    return bwt_summary


def run_posthoc_stage_probes(args, mode, sigma, seed, mnist_train, mnist_test,
                             test_picked, class_order, run_dir, stage_state_history,
                             T_on, T_off, T_skip, device):
    sample_img, _ = mnist_train[0]
    probe_model = build_fresh_model(args, mode, sigma, seed, device)
    probe_model.build(sample_img.unsqueeze(0).to(device))

    stage_eval_history = []
    bwt_matrices = {
        name: np.full((len(class_order), 10), np.nan, dtype=np.float32)
        for name in READOUT_METHODS
    }

    for task_idx, cls in enumerate(class_order):
        stage_state_path = os.path.join(run_dir, stage_state_history[task_idx]['stage_state_path'])
        state = torch.load(stage_state_path, map_location='cpu')
        apply_complete_model_state(probe_model, state, device)
        neurons_labels = state['neurons_labels'].to(device)
        if int(probe_model.RSNN_Exc_pre_spikes.shape[0]) != 1:
            raise RuntimeError(f"Stage state {stage_state_path} was not saved with batch_size=1.")

        seen_classes, stage_test_indices = build_eval_subset(test_picked, class_order, task_idx)
        stage_loader = DataLoader(
            Subset(mnist_test, stage_test_indices),
            batch_size=1,
            shuffle=False,
        )
        probe_payload, stage_eval = probe_dataset(
            probe_model,
            stage_loader,
            stage_test_indices,
            neurons_labels,
            T_on,
            T_off,
            T_skip,
            device,
            desc=f"  Posthoc test after task {task_idx}",
            store_voltage=False,
        )
        stage_path = save_stage_probe(
            run_dir, task_idx, cls, seen_classes, probe_payload, stage_eval
        )

        stage_record = {
            'task_idx': int(task_idx),
            'class_learned': int(cls),
            'seen_classes': [int(c) for c in seen_classes],
            'stage_state_path': stage_state_history[task_idx]['stage_state_path'],
            'stage_representation_path': os.path.relpath(stage_path, run_dir),
        }
        for method in READOUT_METHODS:
            acc, per_cls, _ = stage_eval[method]
            stage_record[f'acc_{method}'] = float(acc)
            stage_record[f'per_class_{method}'] = {
                str(k): float(v) for k, v in per_cls.items()
            }
            for class_id, class_acc in per_cls.items():
                bwt_matrices[method][task_idx, int(class_id)] = float(class_acc)
        stage_eval_history.append(stage_record)
        print(f"  Posthoc stage [{cls}] selective acc: "
              f"{stage_eval['selective'][0]*100:.2f}% on classes {seen_classes}")

    del probe_model
    if device != 'cpu':
        torch.cuda.empty_cache()

    return stage_eval_history, build_bwt_summary(
        class_order, stage_eval_history, bwt_matrices
    )


# ─────────────────────────────────────────────────
#  Core: single run
# ─────────────────────────────────────────────────

def run_single(args, mode, sigma, seed, mnist_train, mnist_test,
               train_picked, test_picked, class_order, order_tag):
    device = args.device
    set_seed(seed)

    cond_name = f"{mode}_sigma{sigma}" if mode == 'distance' else 'random'
    run_name = f"{cond_name}_order{order_tag}_seed{seed}"
    run_dir = os.path.join(args.save_dir, 'runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Run: {run_name}")
    print(f"  σ={sigma}, mode={mode}, seed={seed}")
    print(f"  train/class={args.per_class}, test/class={args.test_per_class}")
    print(f"{'#'*60}")

    # Rebuild data subsets (Subsets share indices; we re-wrap here so training
    # loader iterates in the correct order per task and test loader iterates all
    # classes sequentially).
    train_subsets = {cls: Subset(mnist_train, train_picked[cls]) for cls in range(10)}
    test_indices = []
    test_labels_gt = []
    for cls in range(10):
        test_indices.extend(test_picked[cls])
        test_labels_gt.extend([cls] * len(test_picked[cls]))
    test_loader = DataLoader(
        Subset(mnist_test, test_indices),
        batch_size=1,
        shuffle=False,
    )

    # Build model
    set_seed(seed)
    model = Brain_Like_RSNN(
        inputs_size=torch.Size([1, 1, 28, 28]),
        exc_neurons=4096, inh_neurons=1024,
        sigma=sigma, dt=0.1, mode=mode,
        input_mode=getattr(args, 'input_mode', 'spatial'),
        target_mean=args.target_mean, seed=seed, device=device,
    ).to(device)

    sample_img, _ = train_subsets[class_order[0]][0]
    model.build(sample_img.unsqueeze(0).to(device))
    model.save_checkpoint(os.path.join(run_dir, 'checkpoints', 'init'), tag='init')

    # Burn-in (same recipe as cl_sigma_sweep)
    all_train_indices = []
    for cls in class_order:
        all_train_indices.extend(train_picked[cls])
    burn_loader = DataLoader(Subset(mnist_train, all_train_indices), batch_size=1, shuffle=False)
    model.burn_in(burn_loader, T_on=args.burn_in_on, T_off=args.burn_in_off)

    # Allocate trajectory buffers
    N_exc = model.exc_neurons
    N_inh = model.inh_neurons
    H_dim = N_exc + N_inh
    N_train = args.per_class * 10
    N_test = args.test_per_class * 10

    H_train_rate = torch.zeros((N_train, H_dim), dtype=torch.float32)
    H_train_V = torch.zeros((N_train, H_dim), dtype=torch.float32)
    train_preact = torch.zeros((N_train, 10), dtype=torch.float32)
    train_labels = torch.zeros(N_train, dtype=torch.long)
    train_task_map = torch.zeros(N_train, dtype=torch.long)
    train_step_in_task = torch.zeros(N_train, dtype=torch.long)

    neurons_labels = torch.zeros((N_exc, 10), device=device)
    # snapshot at pre-training (all zeros) + after each task
    nl_snapshots = torch.zeros((11, N_exc, 10), dtype=torch.float32)
    nl_snapshots[0] = neurons_labels.cpu()
    stage_state_history = []

    T_on = args.T_on
    T_off = args.T_off
    T_skip = args.T_skip
    T_rate = (T_on - T_skip) + T_off  # denominator for mean firing rate
    T_V = T_rate                      # matches rate window (V averaged over same window)

    global_step = 0

    # ──── Training ────
    model.enable_plasticity()
    for task_idx, cls in enumerate(class_order):
        subset = train_subsets[cls]
        loader = DataLoader(subset, batch_size=1, shuffle=False)
        print(f"\n  Task {task_idx}: Class {cls} ({len(subset)} samples)")

        with torch.no_grad():
            for step_in_task, (img, label) in enumerate(
                    tqdm(loader, desc=f"  Train cls {cls}", leave=False)):
                x = img.to(device)

                spike_exc = torch.zeros(N_exc, device=device)
                spike_inh = torch.zeros(N_inh, device=device)
                v_exc = torch.zeros(N_exc, device=device)
                v_inh = torch.zeros(N_inh, device=device)

                for t in range(T_on):
                    exc_spikes, inh_spikes = model(x)
                    model.step()
                    if t >= T_skip:
                        spike_exc += exc_spikes[0]
                        spike_inh += inh_spikes[0]
                        v_exc += model.RSNN_Exc_neurons.v[0]
                        v_inh += model.RSNN_Inh_neurons.v[0]
                        neurons_labels[:, cls] += exc_spikes[0]

                zero_input = torch.zeros_like(x)
                for t in range(T_off):
                    exc_spikes, inh_spikes = model(zero_input)
                    model.step()
                    spike_exc += exc_spikes[0]
                    spike_inh += inh_spikes[0]
                    v_exc += model.RSNN_Exc_neurons.v[0]
                    v_inh += model.RSNN_Inh_neurons.v[0]
                    neurons_labels[:, cls] += exc_spikes[0]

                rate_exc = (spike_exc / T_rate).detach().cpu()
                rate_inh = (spike_inh / T_rate).detach().cpu()
                meanV_exc = (v_exc / T_V).detach().cpu()
                meanV_inh = (v_inh / T_V).detach().cpu()

                H_train_rate[global_step, :N_exc] = rate_exc
                H_train_rate[global_step, N_exc:] = rate_inh
                H_train_V[global_step, :N_exc] = meanV_exc
                H_train_V[global_step, N_exc:] = meanV_inh

                # Readout pre-activation with CURRENT neurons_labels (post-update)
                row_mean = neurons_labels.mean(dim=1, keepdim=True)
                selective = (neurons_labels - row_mean).detach()
                preact = torch.matmul(rate_exc.to(device), selective).detach().cpu()
                train_preact[global_step] = preact

                train_labels[global_step] = int(label.item())
                train_task_map[global_step] = task_idx
                train_step_in_task[global_step] = step_in_task
                global_step += 1

        nl_snapshots[task_idx + 1] = neurons_labels.detach().cpu()
        stage_state_path = save_complete_model_state(
            os.path.join(
                run_dir,
                'stage_checkpoints',
                f'task_{task_idx:02d}_class_{cls}_complete_state.pt',
            ),
            model,
            neurons_labels,
            task_idx=task_idx,
            class_learned=cls,
        )
        stage_state_record = {
            'task_idx': int(task_idx),
            'class_learned': int(cls),
            'stage_state_path': os.path.relpath(stage_state_path, run_dir),
        }
        stage_state_history.append(stage_state_record)
        print(f"  Saved complete stage state after class {cls}: "
              f"{stage_state_record['stage_state_path']}")

    # Save trained checkpoint
    model.save_checkpoint(os.path.join(run_dir, 'checkpoints', 'trained'), tag='trained')
    final_state_path = save_complete_model_state(
        os.path.join(run_dir, 'checkpoints', 'trained_complete_state.pt'),
        model,
        neurons_labels,
        task_idx=len(class_order) - 1,
        class_learned=class_order[-1],
    )
    torch.save(neurons_labels.cpu(), os.path.join(run_dir, 'neurons_labels_final.pt'))
    torch.save(nl_snapshots, os.path.join(run_dir, 'neurons_labels_snapshots.pt'))

    torch.save({
        'H_rate': H_train_rate,
        'H_V': H_train_V,
        'preact': train_preact,
        'labels': train_labels,
        'task_map': train_task_map,
        'step_in_task': train_step_in_task,
        'N_exc': N_exc,
        'N_inh': N_inh,
    }, os.path.join(run_dir, 'training_trajectory.pt'))
    print(f"  Saved training_trajectory.pt")

    # ──── Final test-set rate collection + accuracy (plasticity OFF) ────
    final_probe, eval_results = probe_dataset(
        model,
        test_loader,
        test_indices,
        neurons_labels,
        T_on,
        T_off,
        T_skip,
        device,
        desc="  Final test probe",
        store_voltage=True,
    )
    final_probe['source_state_path'] = os.path.relpath(final_state_path, run_dir)
    torch.save(final_probe, os.path.join(run_dir, 'test_final.pt'))
    print(f"  Saved test_final.pt")

    print("\n  Running posthoc stage probes from saved complete states...")
    stage_eval_history, bwt_summary = run_posthoc_stage_probes(
        args,
        mode,
        sigma,
        seed,
        mnist_train,
        mnist_test,
        test_picked,
        class_order,
        run_dir,
        stage_state_history,
        T_on,
        T_off,
        T_skip,
        device,
    )

    result = {
        'run_name': run_name,
        'condition': cond_name,
        'mode': mode,
        'sigma': sigma,
        'seed': seed,
        'class_order': class_order,
        'order_tag': order_tag,
        'per_class_train': args.per_class,
        'per_class_test': args.test_per_class,
        'batch_size_train': 1,
        'batch_size_test': 1,
        'stage_state_history': stage_state_history,
        'stage_eval_history': stage_eval_history,
        'stage_representations_dir': 'stage_representations',
        'stage_checkpoints_dir': 'stage_checkpoints',
        'trained_complete_state_path': os.path.relpath(final_state_path, run_dir),
    }
    print(f"\n  Final accuracy by readout:")
    for method in READOUT_METHODS:
        acc, per_cls, confusion = eval_results[method]
        result[f'final_acc_{method}'] = acc
        result[f'final_per_class_{method}'] = {str(k): v for k, v in per_cls.items()}
        result[f'confusion_{method}'] = confusion.tolist()
        print(f"    [{method}] {acc*100:.2f}%")
    result['final_acc'] = result['final_acc_selective']
    result['final_per_class'] = result['final_per_class_selective']
    result['confusion'] = result['confusion_selective']

    with open(os.path.join(run_dir, 'bwt_summary.json'), 'w') as f:
        json.dump(bwt_summary, f, indent=2)

    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    del model
    if device != 'cpu':
        torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────

def run_experiment(args):
    if int(getattr(args, 'eval_batch_size', 1)) != 1:
        raise ValueError(
            "--eval_batch_size must be 1 for this experiment. "
            "Training, testing, and saved network states all assume batch size 1."
        )

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    class_order = parse_class_order(args.class_order)
    order_tag = format_order_tag(class_order)

    # Load dataset once
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'fmnist':
        mnist_train = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
        mnist_test = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)
    else:
        mnist_train = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    # Build fixed data subsets once (shared across all runs)
    _, train_picked = build_fixed_subsets(mnist_train, args.per_class, DATA_SEED)
    _, test_picked = build_fixed_subsets(mnist_test, args.test_per_class, DATA_SEED)

    train_flat = flatten_picked_indices(train_picked, class_order)
    test_flat = flatten_picked_indices(test_picked, list(range(10)))
    train_hash = indices_hash(train_flat)
    test_hash = indices_hash(test_flat)
    print(f"Data fixation hashes: train={train_hash}  test={test_hash}")

    meta = {
        'class_order': class_order,
        'order_tag': order_tag,
        'per_class_train': args.per_class,
        'per_class_test': args.test_per_class,
        'batch_size_train': 1,
        'batch_size_test': 1,
        'eval_batch_size': 1,
        'DATA_SEED': DATA_SEED,
        'train_indices_hash': train_hash,
        'test_indices_hash': test_hash,
        'train_picked': {int(k): [int(i) for i in v] for k, v in train_picked.items()},
        'test_picked': {int(k): [int(i) for i in v] for k, v in test_picked.items()},
        'sigmas': args.sigmas,
        'seeds': args.seeds,
        'include_random': not args.no_random,
        'T_on': args.T_on,
        'T_off': args.T_off,
        'T_skip': args.T_skip,
        'burn_in_on': args.burn_in_on,
        'burn_in_off': args.burn_in_off,
        'target_mean': args.target_mean,
        'dataset': args.dataset,
        'input_mode': args.input_mode,
    }
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    conditions = [('distance', s, f'distance_sigma{s}') for s in args.sigmas]
    if not args.no_random:
        conditions.append(('random', 1.0, 'random'))

    all_results = []
    for mode, sigma, cond_name in conditions:
        for seed in args.seeds:
            try:
                result = run_single(
                    args, mode, sigma, seed,
                    mnist_train, mnist_test,
                    train_picked, test_picked,
                    class_order, order_tag,
                )
                all_results.append(result)
            except Exception as exc:
                print(f"[ERROR] {cond_name} seed={seed}: {exc}")
                raise

    # Summary
    print("\n" + "="*72)
    print(f"  Summary (N_runs={len(all_results)})")
    print("="*72)
    print(f"{'Run':<48}  {'selective':>10}")
    for r in all_results:
        print(f"{r['run_name']:<48}  "
              f"{r['final_acc_selective']*100:>9.2f}%")

    with open(os.path.join(save_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved under: {save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description='CL Representational Phase Diagram')
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.4, 0.7],
                   help='Distance-mode sigma values')
    p.add_argument('--no_random', action='store_true',
                   help='Skip random-connectivity condition')
    p.add_argument('--seeds', type=int, nargs='+', default=[43, 44, 46],
                   help='Model seeds (one run per seed per condition)')
    p.add_argument('--class_order', type=str, default='0,1,2,3,4,5,6,7,8,9')

    p.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist'])
    p.add_argument('--input_mode', type=str, default='spatial', choices=['spatial'])
    p.add_argument('--data_root', type=str, default='data',
                   help='Directory where torchvision datasets are downloaded')
    p.add_argument('--target_mean', type=float, default=0.05)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')

    # Training
    p.add_argument('--per_class', type=int, default=50,
                   help='Training samples per class')
    p.add_argument('--T_on', type=int, default=1000)
    p.add_argument('--T_off', type=int, default=2000)
    p.add_argument('--T_skip', type=int, default=100)

    # Testing
    p.add_argument('--test_per_class', type=int, default=50)
    p.add_argument('--eval_batch_size', type=int, default=1,
                   help='Must be 1; kept only for backward-compatible CLI parsing.')

    # Burn-in
    p.add_argument('--burn_in_on', type=int, default=2000)
    p.add_argument('--burn_in_off', type=int, default=3000)

    # Output
    p.add_argument('--save_dir', type=str,
                   default='outputs/phase_diagram')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
