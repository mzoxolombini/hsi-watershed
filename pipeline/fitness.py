from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .evaluation import compute_metrics
from .watershed import DATASET_SEGMENT_BOUNDS, run_watershed_pipeline


@dataclass
class FitnessEvaluator:
    image: np.ndarray
    gt: np.ndarray
    train_mask: np.ndarray
    dataset_name: str
    seed: int

    def evaluate(self, params: Sequence[float]) -> tuple[float, dict]:
        pipeline_out = run_watershed_pipeline(self.image, self.gt, params, self.train_mask, self.seed)
        metrics = compute_metrics(self.gt, pipeline_out["prediction"], eval_mask=self.train_mask)

        f_base = 0.55 * metrics["OA"] + 0.45 * metrics["AA"]

        n_min, n_max = DATASET_SEGMENT_BOUNDS[self.dataset_name]
        n_seg = pipeline_out["n_segments"]
        p_seg = max(0.0, (n_min - n_seg) / n_min) + max(0.0, (n_seg - n_max) / n_max)

        n_total = float(np.sum(self.gt > 0))
        n_lab = float(pipeline_out["n_labeled"])
        p_cov = 1.0 - (n_lab / n_total if n_total > 0 else 0.0)

        fitness = float(f_base - 0.1 * p_seg - 0.2 * p_cov)
        details = {
            "fitness": fitness,
            "f_base": float(f_base),
            "P_seg": float(p_seg),
            "P_cov": float(p_cov),
            "metrics": metrics,
            "pipeline": pipeline_out,
        }
        return fitness, details

    def objective(self, params: Sequence[float]) -> float:
        fit, _ = self.evaluate(params)
        return -fit

    def fitness(self, params: Sequence[float]) -> float:
        fit, _ = self.evaluate(params)
        return fit
