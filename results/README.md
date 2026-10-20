# Compact data behind the paper

These files were extracted from the raw outputs of the runners in `experiments/` (see `REPRODUCE.md`). Accuracies are fractions in [0, 1]. Only the paper's readout, the label-calibrated selective vote, is included.

| File | Rows | Fields | Source output |
|---|---|---|---|
| `benchmark_mnist/runs.json` | 70 runs (7 conditions × 10 seeds) | `condition`, `mode`, `sigma`, `seed`, `final_acc_selective`, `final_per_class_selective`, `confusion_selective` (rows: true class, columns: prediction) | `cl_sigma_sweep` → `runs/*/result.json` |
| `benchmark_fmnist/runs.json` | 20 runs (4 conditions × 5 seeds) | as above | `cl_sigma_sweep --dataset fmnist` |
| `adjudication/run_records.csv` | 35 runs (7 conditions × 5 seeds) | `final_accuracy`; `assembly_overlap` (mean pairwise Jaccard of top-100 class assemblies); `ee_contention`, `input_contention` (mean Jaccard overlap of consecutive tasks' top-4000 E→E / top-2000 input-to-E important synapses); `ee_interference`, `input_interference` (mean absolute rewriting of the earlier task's important synapses by the next task) | `money_plot`, `benchmark_interference` |
| `mechanism/fig2_data.json` | per condition | structural and learned span (mean, population SD, number of runs); consecutive E→E contention; per-run distance-resolved E→E contention profiles with bin edges | `spatial_geography_analysis`, `spatial_geography_analysis_extra`, `benchmark_interference` |
| `similarity_rewriting/pair_records.csv` | 54 trials (3 conditions × 6 pairs × 3 seeds) | `class_a`, `class_b`, `group` (similar/dissimilar), `ee_mean_rewriting` (mean absolute change of class A's top-4000 E→E synapses while learning class B) | `step3_functional_similarity_interference` |
| `similarity_rewriting/pixel_similarity.json` | 6 pairs | cosine similarity between MNIST training-set class-mean images | computed from MNIST |
| `fixed_probe_audit/runs/*.json` | 9 runs (3 conditions × seeds 43, 44, 46) | `selective_vote.accuracy_matrix` (rows: stage after class t is learned, columns: class), BWT, forgetting; `drift_records` (`class`, `task_idx`, `tasks_since_learning`, `cosine_drift` = 1 − cosine similarity of the class-mean response to its value at acquisition) | `evaluate_fixed_probe_forgetting` |
| `readout_ladder/readout_ladder.csv` | 15 runs (3 conditions × seeds 42–46) | old-class accuracy of the fixed selective vote, the recalibrated selective vote, and the refit linear decoder, and the gains between them | `evaluate_refit_linear_decoder`, `analyze_refit_readout_ladder` |
| `scope_screens/*.json` | single seed | CIFAR-10 (grayscale, 28×28) and triplet-only plasticity screens: design and per-condition accuracy | `cl_generalization_screen`, `cl_plasticity_robustness_screen` |

Spans and the pooled consecutive contention in `mechanism/fig2_data.json` were stored as a mean and a population SD (ddof = 0) over `n_runs`. The figure scripts convert them to sample SDs with SD × √(n / (n − 1)).
