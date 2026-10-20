"""Recompute the statistics quoted in the paper text from results/ and print them.

Usage: python -m paper_figures.paper_numbers
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

from paper_figures import fig2_mechanism, fig4_adjudication, fig5_similarity
from paper_figures.common import RESULTS, read_json, sd


def by_condition(runs: list[dict]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for r in runs:
        out[r["condition"]][int(r["seed"])] = r["final_acc_selective"] * 100
    return out


def section4() -> None:
    print("Section 4 / Appendix span table (mean ± sample SD)")
    d = fig2_mechanism.load()
    for key in ("sigma=0.4", "sigma=0.7", "sigma=1.0", "random"):
        s = d["span"][key]
        print(f"  {key:<10} structural {s['mask_mean']:.3f} ± {s['mask_sd']:.3f}  learned {s['learn_mean']:.3f} ± "
              f"{s['learn_sd']:.3f}  reduction {100 * (s['mask_mean'] - s['learn_mean']) / s['mask_mean']:.1f}%")


def section5() -> None:
    print("Section 5 / Table 6: MNIST (mean ± sample SD, n per condition)")
    acc = by_condition(read_json(RESULTS / "benchmark_mnist" / "runs.json"))
    for cond, vals in acc.items():
        v = list(vals.values())
        print(f"  {cond:<20} n={len(v)}  {np.mean(v):.2f} ± {sd(v):.2f}%")
    a, b = list(acc["distance_sigma0.7"].values()), list(acc["distance_sigma1.0"].values())
    print(f"  Welch t-test sigma=0.7 vs sigma=1.0: p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f}")

    print("Appendix: Fashion-MNIST (mean ± sample SD)")
    fm = by_condition(read_json(RESULTS / "benchmark_fmnist" / "runs.json"))
    for cond, vals in fm.items():
        v = list(vals.values())
        print(f"  {cond:<20} n={len(v)}  {np.mean(v):.2f} ± {sd(v):.2f}%")
    seeds = sorted(set(fm["distance_sigma0.7"]) & set(fm["random"]))
    x = np.array([fm["distance_sigma0.7"][s] for s in seeds])
    y = np.array([fm["random"][s] for s in seeds])
    print(f"  paired sigma=0.7 - random over seeds {seeds}: {np.mean(x - y):+.1f} pp, "
          f"paired t-test p = {stats.ttest_rel(x, y).pvalue:.2f}")


def section6() -> None:
    print("Section 6 (n = 35 runs; contention and interference average the E->E and input pathways)")
    records = fig4_adjudication.load_records()
    for key in ("overlap", "contention"):
        st = fig4_adjudication.accuracy_stats(records, key)
        print(f"  {key} vs accuracy: Pearson r={st['pearson_r']:.2f} (p={st['pearson_p']:.3g}); "
              f"Spearman rho={st['spearman_rho']:.2f} (p={st['spearman_p']:.2g})")
        for (mode, sigma, _), (m, s) in zip(fig4_adjudication.CONDITIONS,
                                            fig4_adjudication.condition_summary(records, key)):
            print(f"    {mode:<8} {sigma}: {m:.3f} ± {s:.3f}")
    inter = fig4_adjudication.accuracy_stats(records, "interference")
    x = np.array([r["contention"] for r in records])
    y = np.array([r["interference"] for r in records])
    print(f"  interference vs accuracy: Spearman rho={inter['spearman_rho']:.2f}")
    print(f"  contention vs interference: Pearson r={stats.pearsonr(x, y)[0]:.2f}, "
          f"Spearman rho={stats.spearmanr(x, y)[0]:.2f}")


def section7() -> None:
    print("Section 7 (E->E rewriting; three seeds x three pairs per group)")
    summary = fig5_similarity.summarize(fig5_similarity.load_records())
    for cond, s in summary.items():
        print(f"  {cond:<9} ratio {s['ratio']:.2f}x  Mann-Whitney p={s['mann_whitney_p']:.2g}  "
              f"similar {s['similar']['mean']:.4f}  dissimilar {s['dissimilar']['mean']:.4f}")


def appendix_screens() -> None:
    print("Appendix G scope screens (single seed)")
    for name in ("cifar10_gray28", "plasticity_triplet_only"):
        for r in read_json(RESULTS / "scope_screens" / f"{name}.json")["runs"]:
            print(f"  {name:<24} {r['condition']:<36} {100 * r['final_acc_selective']:.1f}%")


def main() -> None:
    for section in (section4, section5, section6, section7, appendix_screens):
        section()
        print()
    print("Section 8, Table 1, and Figure 6 numbers: python -m paper_figures.fig6_drift_readout")


if __name__ == "__main__":
    main()
