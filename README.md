# Distance-Dependent Connectivity Shapes Continual Learning by Controlling Representational Drift

Code and data for the NeurIPS 2026 paper by Chiu-Chang Cheng, Ching-Lung Hsu, Ya-Ning Chang, and Chao-Hung Wang.

We embed a recurrent spiking network with unsupervised plasticity (4096 excitatory and 1024 inhibitory LIF neurons; triplet STDP, heterosynaptic, transmitter-induced, and inhibitory plasticity, following Zenke et al., 2015) in a 3D unit ball, and vary only the range σ of a Gaussian distance-dependent connectivity kernel against a density-matched random control. On 10-way class-incremental MNIST, an intermediate range (σ ≈ 0.7–1.0) performs best under a fixed, label-calibrated selective-vote readout. The analyses relate this to consecutive-task synaptic contention and to how far population responses drift away from the readout.

## Repository layout

| Path | Contents |
|---|---|
| `main/` | The network: neurons, synapses with distance-dependent or random masks, and plasticity rules |
| `experiments/` | Experiment runners and analyses that produce the raw outputs |
| `paper_figures/` | Scripts that rebuild every data figure, Table 1, and the in-text statistics from `results/` |
| `results/` | Compact data behind each figure and table (about 0.3 MB; see `results/README.md`) |
| `manifests/` | Seeds, data-split indices, and analysis settings for each experiment |

## Installation

Tested with Python 3.10.12 and PyTorch 2.2.2 (CUDA 11.8 build) on Ubuntu with an NVIDIA RTX 4090.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org) if the default wheel does not fit your system. Run all commands from the repository root.

## Rebuild the paper figures (minutes, CPU)

```bash
python -m paper_figures.make_all
```

This writes Figures 2–6, A6, A7, the Fashion-MNIST and drift-supplement figures, and Table 1 to `figures/`, and prints the statistics quoted in the text. Each figure also has its own script (for example `python -m paper_figures.fig3_benchmark`). `REPRODUCE.md` maps every figure and table to its script and input file.

## Rerun the experiments (GPU)

`REPRODUCE.md` lists the command for every experiment, with the seeds in `manifests/`. A ten-class run with 100 training images per class takes about three hours on one RTX 4090; the 70-run MNIST sweep took roughly 200 GPU-hours. Datasets (MNIST, Fashion-MNIST, CIFAR-10) are downloaded through `torchvision` into `data/`, and raw outputs are written to `outputs/`; neither is tracked by git.

### Reproducibility notes

- On our hardware, reruns of the same seed were bit-identical: the 35 runs shared between the benchmark and the snapshot-enabled reruns gave identical accuracies.
- Connectivity masks are cached under `mask/`. Generating a mask consumes random numbers, so whether a mask is already cached changes the initial weights that follow. The screens therefore pre-generate all masks before the measured runs. Different GPUs, drivers, or library versions can also change individual numbers; the comparisons in the paper are statistical, over the seeds in `manifests/`.
- All spreads in the paper and in `paper_figures/` are sample standard deviations (ddof=1).

## Citation

```bibtex
@inproceedings{cheng2026distance,
  title     = {Distance-Dependent Connectivity Shapes Continual Learning by Controlling Representational Drift},
  author    = {Cheng, Chiu-Chang and Hsu, Ching-Lung and Chang, Ya-Ning and Wang, Chao-Hung},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). MNIST, Fashion-MNIST, and CIFAR-10 are subject to their own terms.
