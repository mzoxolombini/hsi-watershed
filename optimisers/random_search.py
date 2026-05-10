from __future__ import annotations

import numpy as np


LOW = np.array([1.0, 0.1, 0.01, 10.0, 0.0], dtype=float)
HIGH = np.array([10.0, 5.0, 0.99, 500.0, 2.0], dtype=float)
INT_IDX = (0, 3, 4)


def _repair(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, LOW, HIGH)
    y[list(INT_IDX)] = np.rint(y[list(INT_IDX)])
    return np.clip(y, LOW, HIGH)


def run_random_search(fitness_fn, seed: int, budget: int = 750) -> dict:
    rng = np.random.default_rng(seed)

    best_params = None
    best_fitness = -np.inf

    for _ in range(budget):
        cand = _repair(rng.uniform(LOW, HIGH))
        fit = float(fitness_fn(cand))
        if fit > best_fitness:
            best_fitness = fit
            best_params = cand.copy()

    return {
        "best_params": best_params,
        "best_fitness": float(best_fitness),
        "evaluations": int(budget),
    }
