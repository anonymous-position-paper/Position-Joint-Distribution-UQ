# Anonymous Reproducibility Repository for Figure 1

This repository is provided as anonymous reproducibility material for the experiments discussed in the rebuttal to the paper "Position: Predictive Uncertainty Is Not Enough -- Joint Distribution for Full Uncertainty Representation". Its purpose is to make the analysis behind Fig. 1 directly inspectable during review.

![Paper Figure 1](<Main Figures/Paper Figure 1.png>)

The main entry point is the notebook `evaluate_and_plot_ensemble_ood.ipynb`. It reproduces the evaluation workflow for deep ensembles on mixed ID/OOD test data, regenerates the Figure 1-style plot, and includes additional visualisations with a finer binning configuration so that the reviewer can inspect whether the qualitative behaviour depends on the grouping choice used in the plot.

More specifically, the notebook loads the CIFAR-10 test data, mixes it with OOD data when required, loads the saved ensemble weights, runs inference for the ensemble seeds used in the paper, saves the corresponding uncertainty arrays, averages the results across seeds, and regenerates the final plot. It also includes examples showing how to reproduce the plots directly from stored result arrays without rerunning inference.

The ![additional plots](<Main Figures>) support the same qualitative conclusion discussed in the paper and in the rebuttal: regions with higher uncertainty are enriched in OOD samples, but ID/OOD overlap still persists, and this behaviour remains visible under finer binning choices rather than appearing only under a single plotting configuration.

## Repository structure

The notebook assumes that `PROJECT_ROOT` points to the root of the repository and that this root contains the `experiments/` directory. A minimal expected structure is:

```text
.
├── evaluate_and_plot_ensemble_ood.ipynb
└── experiments/
    ├── modules/
    ├── saved_models/
    └── results/
```

The notebook imports utilities from `experiments.modules.models`, `experiments.modules.dataset_utils`, `experiments.modules.evaluation_functions`, and `experiments.modules.plotting_functions`.

## How to use the notebook

In the configuration cell, set:

```python
PROJECT_ROOT = Path('./').resolve()
```

This should point to the repository root, that is, the directory containing `experiments/`.

The experiment is controlled through the `CONFIG` dictionary, which specifies the batch size, number of ensemble members, data transformation, OOD fraction, OOD dataset, and, when relevant, the CIFAR10C corruption settings. The notebook evaluates the same five seeds used in the paper:

```python
SEEDS_TO_EVALUATE = [0, 1, 2, 3, 4]
```

Running the main cells loads the trained ensemble weights, performs inference, stores the resulting arrays, averages them across seeds, and then regenerates the Figure 1-style plot.

The saved outputs include:

```text
uncertainty_accuracy_vectors_no_order.npz
accuracy_per_model.npz
config.npy
config.txt
```

The plot is generated from the saved averaged arrays using the parameters `PLOT_NUM_BINS`, `PLOT_WINDOW`, and `PLOT_SORT_BY`. The notebook includes both the standard plotting configuration used for the main figure and an additional configuration with more bins for robustness inspection.

If one wants to inspect the effect of the visual grouping directly, the plotting section can be rerun with:

```python
PLOT_NUM_BINS = 100
PLOT_WINDOW = 10
```

This produces a finer discretisation than the default plotting setup.

## Plot-only usage

The notebook also contains examples showing how to regenerate plots directly from stored result arrays, without rerunning inference. This is useful when the averaged outputs are already available under `experiments/results/...`, including alternative OOD fractions such as 50% or 80% SVHN.

## Dependencies

The notebook uses standard Python scientific packages together with PyTorch. A typical setup is:

```bash
pip install numpy matplotlib torch torchvision notebook
```

CUDA support is optional. If no GPU is available, the notebook falls back to CPU.

## Data, weights, and outputs

The notebook assumes that the trained ensemble weights are present under `experiments/saved_models/...`. For full inference runs, the corresponding datasets must also be available to the dataset-loading utilities. If the saved result arrays are already present, the plot-only sections can be executed without recomputing inference.

Per-seed and averaged outputs are written under `experiments/results/cifar10/`, and generated figures are stored in the corresponding results directory, typically under `figures/bin_<num_bins>/`.

## Note for reviewers

This repository is intended to make the experimental workflow behind the rebuttal directly inspectable. In particular, it allows the reviewer to verify how the ensemble models are loaded, how inference is performed, how the uncertainty arrays are produced, how the ID/OOD markers used in the visualisation are obtained, and how the resulting plots behave under alternative binning choices.

Although the notebook focuses on evaluation and plotting, the repository also includes the code used to train the ensemble models. This means the reviewer can inspect not only the final analysis pipeline, but also the training-side implementation underlying the saved weights and the reproduced results. In that sense, the repository is meant to expose the full practical workflow relevant to Fig. 1, rather than only the last plotting step.

## Scope

This material is provided as an anonymous inspection package for the specific Figure 1-related discussion raised during review. It is not intended to be a polished public release, but rather a compact repository containing the notebook, the trained weights, and the relevant code needed to inspect and reproduce the analysis.
