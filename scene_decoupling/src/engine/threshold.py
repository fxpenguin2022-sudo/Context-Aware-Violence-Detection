from __future__ import annotations

import numpy as np

from .metrics import binary_metrics


def scan_thresholds(y_true: np.ndarray, y_prob: np.ndarray, t_min: float, t_max: float, steps: int):
    vals = np.linspace(t_min, t_max, steps)
    records = [binary_metrics(y_true, y_prob, float(t)) for t in vals]
    best = sorted(records, key=lambda x: (x['f1'], x['acc']), reverse=True)[0]
    return best, records
