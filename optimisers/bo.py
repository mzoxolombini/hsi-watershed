from __future__ import annotations

import numpy as np
from scipy.stats import norm

try:
    from scipy.stats import qmc
except ImportError:  # pragma: no cover
    qmc = None

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

try:
    from skopt import Optimizer
    _HAS_SKOPT = True
except ImportError:  # pragma: no cover
    _HAS_SKOPT = False


LOW = np.array([1.0, 0.1, 0.01, 10.0, 0.0], dtype=float)
HIGH = np.array([10.0, 5.0, 0.99, 500.0, 2.0], dtype=float)
INT_IDX = (0, 3, 4)


def _repair(x: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(x, dtype=float), LOW, HIGH)
    y[list(INT_IDX)] = np.rint(y[list(INT_IDX)])
    return np.clip(y, LOW, HIGH)


def _latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    if qmc is not None:
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        unit = sampler.random(n=n)
    else:
        rng = np.random.default_rng(seed)
        unit = rng.random((n, dim))
    return LOW + unit * (HIGH - LOW)


def _run_manual_bo(fitness_fn, seed: int, budget: int) -> dict:
    rng = np.random.default_rng(seed)
    init_n = 15

    X = []
    y = []
    best_params = None
    best_fitness = -np.inf

    for x in _latin_hypercube(init_n, 5, seed):
        xr = _repair(x)
        fit = float(fitness_fn(xr))
        X.append(x)
        y.append(-fit)
        if fit > best_fitness:
            best_fitness = fit
            best_params = xr.copy()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    for _ in range(budget - init_n):
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(5), nu=2.5) + WhiteKernel(1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True)
        gp.fit(X, y)

        candidates = LOW + rng.random((256, 5)) * (HIGH - LOW)
        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-12)

        best_y = np.min(y)
        z = (best_y - mu) / sigma
        ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        x_next = candidates[int(np.argmax(ei))]

        xr = _repair(x_next)
        fit = float(fitness_fn(xr))
        y_next = -fit

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if fit > best_fitness:
            best_fitness = fit
            best_params = xr.copy()

    return {
        "best_params": best_params,
        "best_fitness": float(best_fitness),
        "evaluations": int(budget),
    }


def run_bo(fitness_fn, seed: int, budget: int = 750) -> dict:
    init_n = 15
    guided_n = 735
    if init_n + guided_n != budget:
        raise ValueError("BO budget must be exactly 15 + 735 = 750.")

    if not _HAS_SKOPT:
        return _run_manual_bo(fitness_fn, seed=seed, budget=budget)

    try:
        rng = np.random.default_rng(seed)
        dimensions = [(LOW[i], HIGH[i]) for i in range(5)]
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",
            acq_func="EI",
            acq_optimizer="sampling",
            random_state=seed,
        )

        best_params = None
        best_fitness = -np.inf

        init_points = _latin_hypercube(init_n, 5, seed)
        for x in init_points:
            xr = _repair(x)
            fit = float(fitness_fn(xr))
            optimizer.tell(list(x), -fit)
            if fit > best_fitness:
                best_fitness = fit
                best_params = xr.copy()

        for _ in range(guided_n):
            x = np.asarray(optimizer.ask(), dtype=float)
            xr = _repair(x)
            fit = float(fitness_fn(xr))
            optimizer.tell(list(x), -fit)

            if fit > best_fitness:
                best_fitness = fit
                best_params = xr.copy()

        return {
            "best_params": best_params,
            "best_fitness": float(best_fitness),
            "evaluations": int(budget),
        }
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return _run_manual_bo(fitness_fn, seed=seed, budget=budget)
