# Reproducing the paper

There are two levels:

1. **Figures and statistics from the shipped data** (`results/`): minutes on a CPU.
2. **Raw experiments**: GPU runs that regenerate the outputs from which `results/` was extracted.

Run every command from the repository root.

## 1. Figures, tables, and in-text statistics

```bash
python -m paper_figures.make_all          # everything below
python -m paper_figures.paper_numbers     # statistics quoted in Sections 4-7 and Appendix G
```

| Paper item | Script | Input in `results/` | Output in `figures/` |
|---|---|---|---|
| Fig. 2, Section 4, span table | `paper_figures.fig2_mechanism` | `mechanism/fig2_data.json` | `fig2_mechanism` |
| Fig. 3, Section 5, Table 6 | `paper_figures.fig3_benchmark` | `benchmark_mnist/runs.json` | `fig3_benchmark` |
| Fashion-MNIST figure | `paper_figures.fig3_benchmark` | `benchmark_fmnist/runs.json` | `figA_fmnist_validation` |
| Fig. 4, Section 6 | `paper_figures.fig4_adjudication` | `adjudication/run_records.csv` | `fig4_adjudication` |
| Fig. A6 (contention vs. interference) | `paper_figures.fig4_adjudication` | `adjudication/run_records.csv` | `figA6_interference_bridge` |
| Fig. 5, Section 7 | `paper_figures.fig5_similarity` | `similarity_rewriting/pair_records.csv` | `fig5_similarity_rewriting` |
| Fig. A7 (per-pair breakdown) | `paper_figures.fig5_similarity` | `similarity_rewriting/*` | `figA7_pair_breakdown` |
| Fig. 6, Section 8 | `paper_figures.fig6_drift_readout` | `fixed_probe_audit/runs/`, `readout_ladder/readout_ladder.csv` | `fig6_drift_readout` |
| Table 1 | `paper_figures.fig6_drift_readout` | `fixed_probe_audit/runs/` | `tab_acquisition_retention.tex` |
| Drift and readout supplement | `paper_figures.fig6_drift_readout` | same as Fig. 6 | `figA_drift_readout_supplement` |
| Appendix G scope screens | `paper_figures.paper_numbers` | `scope_screens/*.json` | printed |

Fig. 1 is a schematic and has no data.

## 2. Raw experiments

Shared settings for every run: 4096 excitatory and 1024 inhibitory neurons, target connection density 0.05, class order 0→9, T_on = 1000, T_off = 2000, T_skip = 100 steps, burn-in 2000/3000 steps. Exact seeds and data splits are in `manifests/`.

### 2.1 MNIST benchmark (Fig. 3, Table 6) and Fashion-MNIST

Seeds: σ = 0.4 uses 42–51; every other condition uses 42–49, 58, and 59 (`manifests/benchmark_mnist.json`).

```bash
python -m experiments.cl_sigma_sweep --sigmas 0.5 0.7 1.0 1.3 2.0 --per_class 100 --test_per_class 100 \
    --seed 42 --n_seeds 8 --save_dir outputs/benchmark_mnist
python -m experiments.cl_sigma_sweep --sigmas 0.5 0.7 1.0 1.3 2.0 --per_class 100 --test_per_class 100 \
    --seed 58 --n_seeds 2 --save_dir outputs/benchmark_mnist
python -m experiments.cl_sigma_sweep --sigmas 0.4 --no_random --per_class 100 --test_per_class 100 \
    --seed 42 --n_seeds 10 --save_dir outputs/benchmark_mnist

python -m experiments.cl_sigma_sweep --dataset fmnist --sigmas 0.5 0.7 1.0 --per_class 100 --test_per_class 100 \
    --seed 42 --n_seeds 5 --save_dir outputs/benchmark_fmnist
```

Each run writes `outputs/<experiment>/runs/<condition>_order0123456789_seed<k>/result.json`; `results/benchmark_*/runs.json` holds the selective-vote fields of these files.

### 2.2 Snapshot runs and synaptic analyses (Fig. 2, Fig. 4, Fig. A6)

These reruns save the class-response matrix and weights after every task. Seeds: σ = 0.4 uses 42–51; every other condition uses 47, 48, 49, 58, 59. Fig. 4 and Fig. A6 use σ = 0.4 seeds 42–46 so that every condition has five runs (`manifests/snapshot_runs.json`).

```bash
python -m experiments.cl_sigma_sweep --sigmas 0.5 0.7 1.0 1.3 2.0 --per_class 100 --test_per_class 100 \
    --seed 47 --n_seeds 3 --save_task_snapshots --save_dir outputs/snapshot_runs
python -m experiments.cl_sigma_sweep --sigmas 0.5 0.7 1.0 1.3 2.0 --per_class 100 --test_per_class 100 \
    --seed 58 --n_seeds 2 --save_task_snapshots --save_dir outputs/snapshot_runs
python -m experiments.cl_sigma_sweep --sigmas 0.4 --no_random --per_class 100 --test_per_class 100 \
    --seed 42 --n_seeds 10 --save_task_snapshots --save_dir outputs/snapshot_runs

python -m experiments.money_plot --runs_dir outputs/snapshot_runs/runs --top_k 100 --readout selective \
    --include_random --save_dir outputs/assembly_overlap
python -m experiments.benchmark_interference --runs_dir outputs/snapshot_runs/runs \
    --sigmas 0.4 0.5 0.7 1.0 1.3 2.0 --include_random --task_lag 1 --top_k_ee 4000 --top_k_input 2000 \
    --save_dir outputs/benchmark_interference
python -m experiments.spatial_geography_analysis --runs_dir outputs/snapshot_runs/runs --sigmas 0.4 0.5 0.7 1.0 \
    --save_dir outputs/spatial_geography
python -m experiments.spatial_geography_analysis_extra --runs_dir outputs/snapshot_runs/runs --save_dir outputs/spatial_geography
```

Assembly overlap comes from `money_plot`; contention and directional interference (E→E and input-to-E, averaged in Section 6) come from `benchmark_interference`; spans and distance-resolved contention come from the two `spatial_geography` analyses. In Fig. 2, σ = 0.4 summarizes ten runs (seeds 42–51) and every other condition five. The structural span averages a random sample of 4000 connected synapses per network, drawn without a fixed seed, so recomputed values vary in the third decimal; the learned span uses the deterministic top-4000 important synapses.

### 2.3 Pairwise rewriting assay (Fig. 5, Fig. A7)

```bash
python -m experiments.step3_functional_similarity_interference --save_dir outputs/pairwise_rewriting
```

The defaults are the paper's settings: σ = 0.4, 0.7, and random; seeds 0–2; similar pairs (3,5), (5,8), (3,8); dissimilar pairs (0,1), (1,4), (0,7); 10 training images per class (`manifests/pairwise_rewriting.json`).

### 2.4 Drift and readout audits (Fig. 6, Table 1, drift supplement)

```bash
python -m experiments.cl_phase_diagram --sigmas 0.4 0.7 --seeds 42 43 44 45 46 \
    --per_class 50 --test_per_class 50 --save_dir outputs/phase_diagram
python -m experiments.evaluate_fixed_probe_forgetting --source_dir outputs/phase_diagram \
    --seeds 43 44 46 --samples_per_class 20 --save_dir outputs/fixed_probe_audit
python -m experiments.evaluate_refit_linear_decoder --source_dir outputs/phase_diagram \
    --seeds 42 43 44 45 46 --save_features --save_dir outputs/readout_ladder
python -m experiments.analyze_refit_readout_ladder --aggregate_dir outputs/readout_ladder \
    --feature_dirs outputs/readout_ladder/features --seeds 42 43 44 45 46
```

The phase-diagram runs use a fixed data split (`DATA_SEED = 777`; indices in `manifests/phase_diagram.json`). The fixed probe uses the first 20 test images per class of that split. The readout ladder uses 30 calibration and 20 disjoint evaluation images per class (`manifests/readout_ladder.json`).

### 2.5 Scope screens (Appendix G)

```bash
python -m experiments.cl_generalization_screen --dataset cifar10_gray28 --sigmas 0.4 0.7 \
    --per_class 10 --test_per_class 10 --seed 80 --n_seeds 1
python -m experiments.cl_plasticity_robustness_screen --sigma 0.7 --per_class 10 --test_per_class 10 --seed 80 --n_seeds 1
```

Both screens reset network state before every test image and save no dense checkpoints.
