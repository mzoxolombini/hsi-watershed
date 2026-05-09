from __future__ import annotations

import numpy as np


LOW = np.array([1.0, 0.1, 0.01, 10.0, 0.0], dtype=float)
HIGH = np.array([10.0, 5.0, 0.99, 500.0, 2.0], dtype=float)
RANGE = HIGH - LOW
INT_IDX = (0, 3, 4)


def _repair(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, LOW, HIGH)
    y[list(INT_IDX)] = np.rint(y[list(INT_IDX)])
    return np.clip(y, LOW, HIGH)


def _tournament(pop: np.ndarray, fit: np.ndarray, rng: np.random.Generator, size: int = 3) -> np.ndarray:
    idx = rng.choice(len(pop), size=size, replace=False)
    return pop[idx[np.argmax(fit[idx])]].copy()


def run_standard_ga(fitness_fn, seed: int, budget: int = 750) -> dict:
    population_size = 15
    generations = 50
    elitism = 2
    crossover_prob = 0.8
    mutation_prob = 0.1
    mutation_sigma = 0.1 * RANGE

    if population_size * generations != budget:
        raise ValueError("Standard GA budget must be exactly population * generations (15*50=750).")

    rng = np.random.default_rng(seed)
    population = np.vstack([_repair(rng.uniform(LOW, HIGH)) for _ in range(population_size)])
    fitness = np.array([fitness_fn(ind) for ind in population], dtype=float)
    eval_count = population_size

    best_idx = int(np.argmax(fitness))
    best_params = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    for _ in range(generations - 1):
        elite_idx = np.argsort(fitness)[-elitism:]
        elites = population[elite_idx].copy()

        children = []
        while len(children) < population_size - elitism:
            p1 = _tournament(population, fitness, rng)
            p2 = _tournament(population, fitness, rng)

            if rng.random() < crossover_prob:
                point = int(rng.integers(1, 5))
                c1 = np.concatenate([p1[:point], p2[point:]])
            else:
                c1 = p1.copy()

            for i in range(5):
                if rng.random() < mutation_prob:
                    c1[i] += rng.normal(0.0, mutation_sigma[i])

            children.append(_repair(c1))

        population = np.vstack([elites] + children)
        fitness = np.array([fitness_fn(ind) for ind in population], dtype=float)
        eval_count += population_size

        idx = int(np.argmax(fitness))
        if fitness[idx] > best_fitness:
            best_fitness = float(fitness[idx])
            best_params = population[idx].copy()

    return {
        "best_params": best_params,
        "best_fitness": best_fitness,
        "evaluations": eval_count,
    }
