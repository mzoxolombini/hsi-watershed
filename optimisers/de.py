from __future__ import annotations

import numpy as np


LOW = np.array([1.0, 0.1, 0.01, 10.0, 0.0], dtype=float)
HIGH = np.array([10.0, 5.0, 0.99, 500.0, 2.0], dtype=float)
INT_IDX = (0, 3, 4)


def _repair(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, LOW, HIGH)
    y[list(INT_IDX)] = np.rint(y[list(INT_IDX)])
    return np.clip(y, LOW, HIGH)


def run_de(fitness_fn, seed: int, budget: int = 750) -> dict:
    pop_size = 15
    generations = 50
    F = 0.8
    CR = 0.9

    if pop_size * generations != budget:
        raise ValueError(f"DE budget must be exactly population * generations ({pop_size}*{generations}={pop_size * generations}).")

    rng = np.random.default_rng(seed)

    population = np.vstack([_repair(rng.uniform(LOW, HIGH)) for _ in range(pop_size)])
    fitness = np.array([fitness_fn(ind) for ind in population], dtype=float)
    eval_count = pop_size

    best_idx = int(np.argmax(fitness))
    best_params = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    for _ in range(generations - 1):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = population[rng.choice(idxs, size=3, replace=False)]
            mutant = a + F * (b - c)

            j_rand = int(rng.integers(0, 5))
            trial = population[i].copy()
            for j in range(5):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            trial = _repair(trial)
            trial_fit = float(fitness_fn(trial))
            eval_count += 1

            if trial_fit >= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fit

                if trial_fit > best_fitness:
                    best_fitness = trial_fit
                    best_params = trial.copy()

    return {
        "best_params": best_params,
        "best_fitness": best_fitness,
        "evaluations": eval_count,
    }
