from __future__ import annotations

import numpy as np


LOW = np.array([1.0, 0.1, 0.01, 10.0, 0.0], dtype=float)
HIGH = np.array([10.0, 5.0, 0.99, 500.0, 2.0], dtype=float)
INT_IDX = (0, 3, 4)


def _repair(chrom: np.ndarray) -> np.ndarray:
    g = chrom.astype(float).copy()

    # Rule 1
    if g[0] <= 3 and g[1] > 1.5:
        g[1] = 1.5
    if g[0] >= 7 and g[1] < 1.0:
        g[1] = 1.0

    # Rule 2
    if g[2] < 0.2 and g[3] < 150:
        g[3] = 150
    if g[2] > 0.6 and g[3] > 300:
        g[3] = 300

    # Rule 3
    if g[0] > 5 and int(round(g[4])) == 2:
        g[4] = 0

    g = np.clip(g, LOW, HIGH)
    g[list(INT_IDX)] = np.rint(g[list(INT_IDX)])
    g = np.clip(g, LOW, HIGH)
    return g


def _random_chromosome(rng: np.random.Generator) -> np.ndarray:
    c = rng.uniform(LOW, HIGH)
    c[list(INT_IDX)] = np.rint(c[list(INT_IDX)])
    return _repair(c)


def _tournament(pop: np.ndarray, fit: np.ndarray, rng: np.random.Generator, size: int = 3) -> np.ndarray:
    idx = rng.choice(len(pop), size=size, replace=False)
    return pop[idx[np.argmax(fit[idx])]].copy()


def _block_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    child = np.empty_like(a)
    child[0:2] = a[0:2] if rng.random() < 0.5 else b[0:2]
    child[2:4] = a[2:4] if rng.random() < 0.5 else b[2:4]
    child[4] = a[4] if rng.random() < 0.5 else b[4]
    return _repair(child)


def _coupled_mutation(c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = c.astype(float).copy()

    if rng.random() < 0.3:
        dk = rng.choice([-1, 0, 1])
        m[0] = m[0] + dk
        m[1] = m[1] + 0.4 * dk + rng.normal(0.0, 0.15)

    if rng.random() < 0.3:
        dg = rng.normal(0.0, 0.08)
        m[2] = m[2] + 0.25 * dg
        m[3] = m[3] - 80.0 * dg

    if rng.random() < 0.15:
        m[4] = m[4] + rng.choice([-1, 0, 1])

    return _repair(m)


def run_dae_ga(fitness_fn, seed: int, budget: int = 750) -> dict:
    population_size = 15
    generations = 50
    elitism = 2
    crossover_prob = 0.8

    if population_size * generations != budget:
        raise ValueError("DAE-GA budget must be exactly population * generations (15*50=750).")

    rng = np.random.default_rng(seed)
    population = np.vstack([_random_chromosome(rng) for _ in range(population_size)])

    fitness = np.array([fitness_fn(ind) for ind in population], dtype=float)
    eval_count = population_size

    best_idx = int(np.argmax(fitness))
    best_params = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    for _ in range(generations - 1):
        elite_idx = np.argsort(fitness)[-elitism:]
        elites = population[elite_idx].copy()

        offspring = []
        while len(offspring) < population_size - elitism:
            pa = _tournament(population, fitness, rng)
            pb = _tournament(population, fitness, rng)

            if rng.random() < crossover_prob:
                child = _block_crossover(pa, pb, rng)
            else:
                child = _repair(pa if rng.random() < 0.5 else pb)

            child = _coupled_mutation(child, rng)
            offspring.append(child)

        population = np.vstack([elites] + offspring)
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
