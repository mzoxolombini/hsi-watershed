# GA-Optimised Watershed Segmentation for Hyperspectral Images

A genetic algorithm (GA) optimised watershed segmentation pipeline for hyperspectral remote sensing datasets. The pipeline automatically tunes watershed and classifier parameters using a GA, then evaluates performance with standard classification metrics.

---

## Overview

This project combines:
- **PCA-based dimensionality reduction** for hyperspectral imagery
- **Watershed segmentation** with marker-based region growing
- **Genetic Algorithm optimisation** (via [PyGAD](https://pygad.readthedocs.io/)) to search for optimal segmentation parameters
- **Local search refinement** to fine-tune the GA's best solution
- **Multiple classifiers** (Random Forest, SVM, KNN) for pixel-level labelling
- **Confidence-based prediction refinement** to improve boundary quality

Supported datasets (auto-downloaded):
| Dataset | Classes | Spatial Size |
|---|---|---|
| Indian Pines | 16 | 145 × 145 |
| Salinas | 16 | 512 × 217 |
| Pavia University | 9 | 610 × 340 |

---

## Requirements

- Python 3.8+
- PyQt5 *(optional — enables interactive plot windows)*

Install dependencies:

```bash
pip install numpy scipy scikit-image scikit-learn matplotlib seaborn joblib pygad PyQt5
```

---

## Project Structure

```
watershedSegmentation/
├── GAOptimisedWatershedSegmentation.py   # Main pipeline
├── data/                                 # Auto-created; stores downloaded .mat files
├── watershed_cache/                      # Auto-created; joblib cache (cleared on each run)
└── README.md
```

---

## Usage

Run the full pipeline across all three datasets:

```bash
python GAOptimisedWatershedSegmentation.py
```

Datasets are downloaded automatically on the first run into the `data/` directory.

### Output

- **Console** — per-dataset metrics (OA, AA, Kappa, Dice, IoU) and GA progress logs
- **Plot window** — RGB composite, ground truth, and segmentation result side by side (requires PyQt5)
- **PNG file** — saved as `<dataset>_results.png` when running in non-interactive mode

---

## Configuration

All key parameters are in the `CONFIG` dictionary at the top of the script:

```python
CONFIG = {
    'datasets': ['IndianPines', 'Salinas', 'PaviaU'],
    'ga_params': {
        'population_size': 15,
        'generations': 50,
        'parents_mating': 5,
        'mutation_percent_genes': 20,
        'saturate_generations': 10,   # Early stopping patience
        'timeout': 14400,             # Seconds
    },
    'downsample_factor': 2,           # Set to 1 to use full resolution
    'watershed_params': {
        'conf_thresh': 0.60,
        'dilation_radius': 3
    },
    'classifiers': {
        'RF': {'n_estimators': 50, 'random_state': 42, 'n_jobs': -1},
        'SVM': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True},
        'KNN': {'n_neighbors': 5, 'n_jobs': -1}
    }
}
```

Dataset-specific GA fitness weights (smoothness penalty, target region count, minimum coverage) are in `DATASET_FITNESS_CONFIG`.

---

## GA Gene Encoding

Each GA solution (chromosome) encodes five parameters:

| Gene | Parameter | Range |
|---|---|---|
| 0 | Number of PCA components | 1 – 10 |
| 1 | Gaussian smoothing sigma | 0.1 – 5.0 |
| 2 | Watershed threshold | 0.01 – 0.99 |
| 3 | Minimum segment size (pixels) | 10 – 500 |
| 4 | Classifier index (0=RF, 1=SVM, 2=KNN) | 0 – 2 |

### Fitness Function

```
fitness = 0.55 × OA + 0.45 × AA − coverage_penalty − smoothness_penalty
```

- **Coverage penalty** — applied when fewer than `min_coverage_pct` of ground-truth pixels are labelled  
- **Smoothness penalty** — applied when the number of segments exceeds `target_regions`

---

## Metrics

| Metric | Description |
|---|---|
| OA | Overall Accuracy |
| AA | Average per-class Accuracy |
| Kappa | Cohen's Kappa coefficient |
| Dice | Macro-averaged F1 score |
| IoU | Macro-averaged Jaccard index |

---

## Notes

- The `watershed_cache/` directory is cleared at the start of each run and removed on completion. If the run is interrupted, delete it manually.
- On Windows, timeout handling uses a background thread instead of POSIX signals.
- Downsampling (`downsample_factor: 2`) is enabled by default to reduce runtime. Set it to `1` for full-resolution results.
