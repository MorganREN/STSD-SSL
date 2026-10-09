"""
Segmentation metrics computed from saved predictions.

Fold-level scores accumulate one confusion matrix over all test points of the fold
(micro-averaged over scenes); classes absent from both prediction and ground truth
are excluded from the mean.
"""

import numpy as np
from scipy.spatial import cKDTree

from stsd.utils import CLASS_NAMES, RARE_CLASSES

NUM_CLASSES = len(CLASS_NAMES)


def confusion(pred, gt, num_classes=NUM_CLASSES):
    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        gt[valid].astype(np.int64) * num_classes + pred[valid], minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def scores(conf):
    """Per-class IoU / accuracy and summary scores from a confusion matrix (rows = gt)."""
    tp = np.diag(conf).astype(np.float64)
    gt_count, pred_count = conf.sum(1), conf.sum(0)
    union = gt_count + pred_count - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    acc = np.where(gt_count > 0, tp / np.maximum(gt_count, 1), np.nan)
    rare = [CLASS_NAMES.index(c) for c in RARE_CLASSES]
    return dict(
        miou=np.nanmean(iou),
        miou_rare=np.nanmean(iou[rare]),
        macc=np.nanmean(acc),
        oa=tp.sum() / max(conf.sum(), 1),
        iou=dict(zip(CLASS_NAMES, iou)),
    )


def boundary_f1(coord, pred, gt, cls, tol=0.02):
    """
    Tolerance F1 for thin structures (joints, bolt holes): a predicted `cls` point is
    correct if a ground-truth `cls` point lies within `tol` metres, and vice versa.
    Returns (precision, recall, f1); NaN if the class is absent from both.
    """
    p, g = coord[pred == cls], coord[gt == cls]
    if len(p) == 0 and len(g) == 0:
        return np.nan, np.nan, np.nan
    if len(p) == 0 or len(g) == 0:
        return 0.0, 0.0, 0.0
    precision = np.mean(cKDTree(g).query(p, distance_upper_bound=tol)[0] <= tol)
    recall = np.mean(cKDTree(p).query(g, distance_upper_bound=tol)[0] <= tol)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def voxel_majority(values, inverse, num_voxels, num_classes=NUM_CLASSES):
    """Majority label per voxel from per-original-point labels (`inverse`: point -> voxel)."""
    votes = np.zeros((num_voxels, num_classes), np.int32)
    np.add.at(votes, (inverse, values), 1)
    return votes.argmax(1)


def aulc(budgets, values, all_budget):
    """Area under the score-vs-log2(budget) curve, normalised by the log2 range."""
    budgets = np.array([all_budget if b in (None, "all") else b for b in budgets], np.float64)
    order = np.argsort(budgets)
    x, y = np.log2(budgets[order]), np.asarray(values, np.float64)[order]
    return np.trapz(y, x) / (x[-1] - x[0])
