from __future__ import annotations

from collections import defaultdict

import numpy as np

from optimisers import run_bo, run_dae_ga, run_de, run_pso, run_random_search, run_standard_ga
from pipeline import FitnessEvaluator, compute_metrics, load_dataset, normalise_image, run_watershed_pipeline, stratified_train_test_split


DATASETS = ["Indian Pines", "Salinas", "Pavia University"]
SEEDS = list(range(42, 52))
BUDGET = 750
TRAIN_RATIO = 0.30

OPTIMISERS = {
    "DAE-GA": run_dae_ga,
    "Standard GA": run_standard_ga,
    "PSO": run_pso,
    "DE": run_de,
    "BO": run_bo,
    "Random Search": run_random_search,
}

METRICS = ["OA", "AA", "Kappa", "Dice", "IoU"]


def _format_metric(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def run_experiment() -> None:
    all_results: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset in DATASETS:
        image, gt = load_dataset(dataset, data_dir="data")
        image = normalise_image(image)

        for seed in SEEDS:
            train_mask, test_mask = stratified_train_test_split(gt, train_ratio=TRAIN_RATIO, seed=seed)
            evaluator = FitnessEvaluator(
                image=image,
                gt=gt,
                train_mask=train_mask,
                dataset_name=dataset,
                seed=seed,
            )
            def fitness_fn(params):
                return evaluator.evaluate(params)[0]

            for name, optimiser in OPTIMISERS.items():
                best = optimiser(fitness_fn, seed=seed, budget=BUDGET)
                pipeline_out = run_watershed_pipeline(image, gt, best["best_params"], train_mask=train_mask, seed=seed)
                metrics = compute_metrics(gt, pipeline_out["prediction"], eval_mask=test_mask)

                for metric_name in METRICS:
                    all_results[dataset][name][metric_name].append(metrics[metric_name])

                print(
                    f"[{dataset}] seed={seed} optimiser={name}: "
                    + ", ".join(f"{m}={metrics[m]:.4f}" for m in METRICS)
                    + f", evals={best['evaluations']}"
                )

    print("\n=== Summary (mean ± std over 10 seeds) ===")
    for dataset in DATASETS:
        print(f"\nDataset: {dataset}")
        header = "Optimiser".ljust(16) + " | " + " | ".join(m.ljust(14) for m in METRICS)
        print(header)
        print("-" * len(header))
        for name in OPTIMISERS:
            row = [name.ljust(16)]
            for metric_name in METRICS:
                row.append(_format_metric(all_results[dataset][name][metric_name]).ljust(14))
            print(" | ".join(row))


if __name__ == "__main__":
    run_experiment()
