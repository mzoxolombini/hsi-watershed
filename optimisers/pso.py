from __future__ import annotations

import numpy as np


LOW = np.array([1.0, 0.1, 0.01, 10.0, 0.0], dtype=float)
HIGH = np.array([10.0, 5.0, 0.99, 500.0, 2.0], dtype=float)
RANGE = HIGH - LOW
INT_IDX = (0, 3, 4)


def _round_discrete(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, LOW, HIGH).copy()
    y[list(INT_IDX)] = np.rint(y[list(INT_IDX)])
    return np.clip(y, LOW, HIGH)


def run_pso(fitness_fn, seed: int, budget: int = 750) -> dict:
    n_particles = 15
    iterations = 50
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    if n_particles * iterations != budget:
        raise ValueError("PSO budget must be exactly particles * iterations (15*50=750).")

    rng = np.random.default_rng(seed)

    pos = rng.uniform(LOW, HIGH, size=(n_particles, 5))
    vel_max = 0.2 * RANGE
    vel = rng.uniform(-vel_max, vel_max, size=(n_particles, 5))

    pbest_pos = pos.copy()
    pbest_fit = np.full(n_particles, -np.inf)
    gbest_pos = pos[0].copy()
    gbest_fit = -np.inf

    eval_count = 0

    for _ in range(iterations):
        for i in range(n_particles):
            candidate = _round_discrete(pos[i])
            fit = float(fitness_fn(candidate))
            eval_count += 1

            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = candidate
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = candidate.copy()

        r1 = rng.random(size=(n_particles, 5))
        r2 = rng.random(size=(n_particles, 5))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        vel = np.clip(vel, -vel_max, vel_max)
        pos = np.clip(pos + vel, LOW, HIGH)

    return {
        "best_params": gbest_pos,
        "best_fitness": float(gbest_fit),
        "evaluations": eval_count,
    }
