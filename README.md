# Optimisation of Watershed Segmentation for Hyperspectral Imagery

Implementation of the **Dependency-Aware Encoding Genetic Algorithm (DAE-GA)** proposed in:

> Mbini, M. & Nyathi, T. (2026). *Optimisation of Watershed Segmentation for Hyperspectral Imagery*.

## Method

The DAE-GA uses a dependency-aware chromosome with three blocks:

- **Block 1:** `(k, σ)` spectral dimensionality and smoothing scale
- **Block 2:** `(t, s)` flooding threshold and region size
- **Block 3:** `c` classifier choice

The full 4-stage segmentation pipeline is:

1. **PCA** to `k` components
2. **Gradient** from Gaussian-smoothed first PC using Sobel
3. **Watershed** using marker extraction on `-gradient`, threshold control `t`, and small-region merging with `s`
4. **Classification** by RF/SVM/kNN and majority vote per segment

DAE-GA operators:

- **Block crossover** over `(k,σ)`, `(t,s)`, and `c`
- **Repair operator** with dependency rules for spectral-scale, threshold-size, and dimensionality-classifier consistency
- **Semantically coupled mutation** in three probabilistic modes

## Parameter Space

The optimised vector is `p = (k, σ, t, s, c)`:

- `k ∈ [1, 10]` (integer)
- `σ ∈ [0.1, 5.0]`
- `t ∈ [0.01, 0.99]`
- `s ∈ [10, 500]` (integer)
- `c ∈ {0, 1, 2}` with `0=RF, 1=SVM, 2=kNN`

## Datasets

- Indian Pines (145×145, 200 bands, 16 classes)
- Salinas (512×217, 204 bands, 16 classes)
- Pavia University (610×340, 103 bands, 9 classes)

## Installation

```bash
pip install numpy scipy scikit-image scikit-learn matplotlib seaborn joblib scikit-optimize
```

## Usage

```bash
python main.py
```

Runs all six optimisers on all three datasets for 10 seeds (42–51), with budget 750 evaluations per run.

## Results (mean OA % over 10 runs)

The script prints summary tables (OA, AA, Kappa, Dice, IoU) in the format `mean ± std` for each dataset and optimiser, matching the paper’s table-style reporting.
