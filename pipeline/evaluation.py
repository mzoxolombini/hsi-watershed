from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


def compute_metrics(gt: np.ndarray, pred: np.ndarray, eval_mask: np.ndarray | None = None) -> dict:
    if eval_mask is None:
        eval_mask = gt > 0

    valid = eval_mask & (gt > 0)
    if not np.any(valid):
        return {"OA": 0.0, "AA": 0.0, "Kappa": 0.0, "Dice": 0.0, "IoU": 0.0}

    y_true = gt[valid]
    y_pred = pred[valid]

    oa = float(np.mean(y_true == y_pred))
    classes = np.unique(y_true)
    class_acc = []
    for cls in classes:
        cls_mask = y_true == cls
        class_acc.append(float(np.mean(y_pred[cls_mask] == cls)))
    aa = float(np.mean(class_acc)) if class_acc else 0.0

    kappa = float(cohen_kappa_score(y_true, y_pred, labels=classes)) if classes.size > 1 else 0.0
    dice = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    iou = float(dice / (2.0 - dice)) if dice < 2.0 else 0.0

    return {"OA": oa, "AA": aa, "Kappa": kappa, "Dice": dice, "IoU": iou}
