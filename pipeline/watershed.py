from __future__ import annotations

from pathlib import Path
from typing import Sequence
import urllib.request

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat
from skimage.feature import peak_local_max
from skimage.filters import gaussian, sobel
from skimage.segmentation import watershed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

CLASSIFIER_NAMES = {0: "RF", 1: "SVM", 2: "kNN"}

PARAM_BOUNDS = {
    "k": (1, 10),
    "sigma": (0.1, 5.0),
    "t": (0.01, 0.99),
    "s": (10, 500),
    "c": (0, 2),
}

DATASET_SEGMENT_BOUNDS = {
    "Indian Pines": (80, 400),
    "Salinas": (300, 1500),
    "Pavia University": (200, 1000),
}

DATASET_FILES = {
    "Indian Pines": {
        "data_file": "IndianPines_corrected.mat",
        "gt_file": "IndianPines_gt.mat",
        "data_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "data_url": "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "gt_url": "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    },
    "Salinas": {
        "data_file": "Salinas_corrected.mat",
        "gt_file": "Salinas_gt.mat",
        "data_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "data_url": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        "gt_url": "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
    },
    "Pavia University": {
        "data_file": "PaviaU_corrected.mat",
        "gt_file": "PaviaU_gt.mat",
        "data_key": "paviaU",
        "gt_key": "paviaU_gt",
        "data_url": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "gt_url": "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
    },
}


def load_dataset(dataset_name: str, data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    spec = DATASET_FILES[dataset_name]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / spec["data_file"]
    gt_path = data_dir / spec["gt_file"]

    if not data_path.exists():
        urllib.request.urlretrieve(spec["data_url"], data_path)
    if not gt_path.exists():
        urllib.request.urlretrieve(spec["gt_url"], gt_path)

    data_mat = loadmat(data_path)
    gt_mat = loadmat(gt_path)

    data_keys = [k for k in data_mat.keys() if not k.startswith("__")]
    gt_keys = [k for k in gt_mat.keys() if not k.startswith("__")]
    if not data_keys or not gt_keys:
        raise KeyError(f"Unable to locate dataset keys for {dataset_name}.")

    data_key = spec["data_key"] if spec["data_key"] in data_mat else data_keys[0]
    gt_key = spec["gt_key"] if spec["gt_key"] in gt_mat else gt_keys[0]

    return data_mat[data_key].astype(np.float32), gt_mat[gt_key].astype(np.int32)


def normalise_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mn = float(np.min(image))
    mx = float(np.max(image))
    if mx <= mn:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mn) / (mx - mn)


def stratified_train_test_split(gt: np.ndarray, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_mask = np.zeros_like(gt, dtype=bool)
    gt_flat = np.ravel(gt, order="C")

    for cls in np.unique(gt):
        if cls == 0:
            continue
        cls_idx = np.flatnonzero(gt_flat == cls)
        rng.shuffle(cls_idx)
        n_train = max(1, int(round(train_ratio * cls_idx.size)))
        train_mask.flat[cls_idx[:n_train]] = True

    labelled = gt > 0
    test_mask = labelled & ~train_mask
    return train_mask, test_mask


def parse_parameters(params: Sequence[float]) -> dict:
    k = int(np.clip(np.rint(params[0]), *PARAM_BOUNDS["k"]))
    sigma = float(np.clip(params[1], *PARAM_BOUNDS["sigma"]))
    t = float(np.clip(params[2], *PARAM_BOUNDS["t"]))
    s = int(np.clip(np.rint(params[3]), *PARAM_BOUNDS["s"]))
    c = int(np.clip(np.rint(params[4]), *PARAM_BOUNDS["c"]))
    return {"k": k, "sigma": sigma, "t": t, "s": s, "c": c}


def _merge_small_regions(labels: np.ndarray, min_size: int, mask: np.ndarray) -> np.ndarray:
    merged = labels.copy()
    ids = np.unique(merged[mask])
    ids = ids[ids > 0]
    for region_id in ids:
        region = merged == region_id
        if int(np.sum(region)) >= min_size:
            continue
        boundary = ndi.binary_dilation(region, iterations=1) & (~region) & mask
        neighbors = merged[boundary]
        neighbors = neighbors[neighbors > 0]
        if neighbors.size:
            merged[region] = int(np.bincount(neighbors).argmax())
    return merged


def _build_classifier(classifier_index: int, seed: int):
    if classifier_index == 0:
        return RandomForestClassifier(n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1)
    if classifier_index == 1:
        return SVC(kernel="rbf", C=100, gamma=0.01)
    if classifier_index == 2:
        return KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    raise ValueError(f"Unknown classifier index: {classifier_index}")


def _majority_vote_segments(pixel_labels: np.ndarray, segments: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pixel_labels, dtype=np.int32)
    segment_ids = np.unique(segments[mask])
    for seg_id in segment_ids:
        if seg_id <= 0:
            continue
        seg_mask = (segments == seg_id) & mask
        votes = pixel_labels[seg_mask]
        votes = votes[votes > 0]
        if votes.size:
            out[seg_mask] = int(np.bincount(votes).argmax())
    return out


def run_watershed_pipeline(
    image: np.ndarray,
    gt: np.ndarray,
    params: Sequence[float],
    train_mask: np.ndarray,
    seed: int,
) -> dict:
    p = parse_parameters(params)
    h, w, b = image.shape
    gt_mask = gt > 0

    pixels = image.reshape(-1, b)
    pca = PCA(n_components=p["k"], random_state=seed)
    reduced = pca.fit_transform(pixels).reshape(h, w, p["k"])

    first_pc = reduced[..., 0]
    smoothed = gaussian(first_pc, sigma=p["sigma"], preserve_range=True)
    gradient = sobel(smoothed)

    inv_grad = -gradient
    valid_inv = inv_grad[gt_mask]
    threshold_abs = float(np.quantile(valid_inv, p["t"])) if valid_inv.size else float(np.min(inv_grad))
    coords = peak_local_max(inv_grad, labels=gt_mask.astype(np.uint8), min_distance=3, threshold_abs=threshold_abs)

    markers = np.zeros((h, w), dtype=np.int32)
    if coords.size:
        markers[coords[:, 0], coords[:, 1]] = np.arange(1, coords.shape[0] + 1)
    else:
        grad_valid = gradient[gt_mask]
        cut = float(np.quantile(grad_valid, p["t"])) if grad_valid.size else float(np.max(gradient))
        fallback = (gradient <= cut) & gt_mask
        markers, _ = ndi.label(fallback)

    if int(np.max(markers)) == 0:
        markers, _ = ndi.label(gt_mask)

    segments = watershed(gradient, markers=markers, mask=gt_mask)
    segments = _merge_small_regions(segments, min_size=p["s"], mask=gt_mask)

    y_train = gt[train_mask]
    x_train = image[train_mask]

    classifier = _build_classifier(p["c"], seed)
    classifier.fit(x_train, y_train)

    pixel_pred = np.zeros((h, w), dtype=np.int32)
    pixel_pred[gt_mask] = classifier.predict(image[gt_mask]).astype(np.int32)

    seg_pred = _majority_vote_segments(pixel_pred, segments, gt_mask)
    n_segments = int(np.unique(segments[gt_mask]).size)
    n_labeled = int(np.sum((seg_pred > 0) & gt_mask))

    return {
        "params": p,
        "segments": segments,
        "prediction": seg_pred,
        "n_segments": n_segments,
        "n_labeled": n_labeled,
    }
