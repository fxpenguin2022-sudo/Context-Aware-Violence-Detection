from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    specificity = tn / (tn + fp + eps)
    npv = tn / (tn + fn + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    balanced_acc = 0.5 * (recall + specificity)

    return {
        'acc': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'npv': float(npv),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'balanced_acc': float(balanced_acc),
        'support_pos': float((y_true == 1).sum()),
        'support_neg': float((y_true == 0).sum()),
        'pred_pos': float((y_pred == 1).sum()),
        'pred_neg': float((y_pred == 0).sum()),
        'tp': float(tp),
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'threshold': float(threshold),
    }
