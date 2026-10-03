"""Image- and pixel-level anomaly detection metrics: AUROC, AP, F1-max and AUPRO.

Two pixel-level protocols are provided for each class:

* pooled: pixels of all test images of the class are pooled before computing the metric
  (the usual MVTec-AD / VisA protocol);
* per-image: metrics are computed on each anomalous image and averaged over those images.

Image-level metrics are the same under both protocols.
"""

from typing import Dict, Sequence

import numpy as np
from skimage import measure
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

IMAGE_METRICS = ("auroc", "ap", "f1_max")
PIXEL_METRICS = ("auroc", "ap", "f1_max", "aupro")


def f1_max(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 score over all thresholds of the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.zeros_like(p)
    np.divide(2 * p * r, denom, out=f1, where=denom > 0)
    return float(f1.max()) if f1.size else 0.0


def aupro(masks: np.ndarray, amaps: np.ndarray, num_thresholds: int = 200, max_fpr: float = 0.3) -> float:
    """Area under the per-region overlap (PRO) curve for FPR in [0, max_fpr], normalized to [0, 1].

    Args:
        masks: binary ground-truth masks, shape [N, H, W].
        amaps: anomaly maps, shape [N, H, W].
        num_thresholds: number of evenly spaced thresholds between amaps.min() and amaps.max().
        max_fpr: integration limit of the false positive rate.

    The PRO at a threshold is the mean, over all connected ground-truth regions, of the
    fraction of the region's pixels above the threshold.
    """
    masks = np.asarray(masks)
    amaps = np.asarray(amaps)
    if masks.ndim != 3 or masks.shape != amaps.shape:
        raise ValueError(f"masks and amaps must both have shape [N, H, W], got {masks.shape} and {amaps.shape}")
    masks = masks > 0
    if not masks.any() or masks.all():
        raise ValueError("masks must contain both normal and anomalous pixels")

    min_th, max_th = amaps.min(), amaps.max()
    if not max_th > min_th:
        return 0.0
    delta = (max_th - min_th) / num_thresholds
    thresholds = np.arange(min_th, max_th, delta)

    def fraction_above(values):
        """Fraction of `values` strictly above each threshold."""
        values = np.sort(values)
        return (values.size - np.searchsorted(values, thresholds, side="right")) / values.size

    fprs = fraction_above(amaps[~masks])
    overlaps = []
    for mask, amap in zip(masks, amaps):
        regions = measure.label(mask)
        for region in range(1, regions.max() + 1):
            overlaps.append(fraction_above(amap[regions == region]))
    pros = np.mean(overlaps, axis=0)

    keep = fprs < max_fpr
    pros, fprs = pros[keep], fprs[keep]
    if fprs.size < 2 or fprs.max() == 0:
        return 0.0
    return float(auc(fprs / fprs.max(), pros))


def image_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    """Image-level AUROC, AP and F1-max (all zeros if only one class is present)."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        return dict.fromkeys(IMAGE_METRICS, 0.0)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1_max": f1_max(labels, scores),
    }


def pixel_metrics(masks: np.ndarray, amaps: np.ndarray, num_thresholds: int = 200) -> Dict[str, float]:
    """Pixel-level AUROC, AP, F1-max and AUPRO over all pixels of masks/amaps.

    Accepts a single image ([H, W]) or a stack of images ([N, H, W]). Returns all zeros
    if the masks are all normal or all anomalous.
    """
    masks = np.asarray(masks) > 0
    amaps = np.asarray(amaps)
    if masks.ndim == 2:
        masks, amaps = masks[None], amaps[None]
    gt = masks.ravel().astype(int)
    scores = amaps.ravel().astype(float)
    if np.unique(gt).size < 2:
        return dict.fromkeys(PIXEL_METRICS, 0.0)
    return {
        "auroc": float(roc_auc_score(gt, scores)),
        "ap": float(average_precision_score(gt, scores)),
        "f1_max": f1_max(gt, scores),
        "aupro": aupro(masks, amaps, num_thresholds),
    }


def class_summary(
    labels: Sequence[int],
    scores: Sequence[float],
    masks: np.ndarray,
    amaps: np.ndarray,
    per_image_pixel: Sequence[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Metrics of one class under the pooled and the per-image protocol.

    Args:
        labels: image labels (0 normal, 1 anomalous).
        scores: image-level anomaly scores.
        masks: ground-truth masks, shape [N, H, W].
        amaps: anomaly heatmaps, shape [N, H, W].
        per_image_pixel: output of ``pixel_metrics`` for each image.

    Returns:
        {"pooled": {...}, "per_image": {...}} with keys image_auroc, ..., pixel_aupro.
    """
    image = {f"image_{k}": v for k, v in image_metrics(labels, scores).items()}
    pooled = {f"pixel_{k}": v for k, v in pixel_metrics(masks, amaps).items()}

    anomalous = [m for m, label in zip(per_image_pixel, labels) if label == 1]
    per_image = {
        f"pixel_{k}": float(np.mean([m[k] for m in anomalous])) if anomalous else 0.0 for k in PIXEL_METRICS
    }
    return {"pooled": {**image, **pooled}, "per_image": {**image, **per_image}}
