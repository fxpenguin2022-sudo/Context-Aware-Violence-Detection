from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scene_decoupling.src.engine.metrics import binary_metrics


class TestMetrics(unittest.TestCase):
    def test_metrics(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_prob = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
        m = binary_metrics(y_true, y_prob, 0.5)
        self.assertAlmostEqual(m['acc'], 1.0, places=6)
        self.assertAlmostEqual(m['f1'], 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
