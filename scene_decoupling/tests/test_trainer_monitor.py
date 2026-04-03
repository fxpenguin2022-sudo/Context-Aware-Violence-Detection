from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scene_decoupling.src.engine.trainer import _monitor_checkpoint_name, _normalize_monitor


class TestTrainerMonitor(unittest.TestCase):
    def test_monitor_normalize(self) -> None:
        self.assertEqual(_normalize_monitor('acc'), 'acc')
        self.assertEqual(_normalize_monitor('accuracy'), 'acc')
        self.assertEqual(_normalize_monitor('val_loss'), 'val_loss')
        self.assertEqual(_normalize_monitor('f1'), 'f1')

    def test_checkpoint_name(self) -> None:
        self.assertEqual(_monitor_checkpoint_name('acc'), 'best_acc.pt')
        self.assertEqual(_monitor_checkpoint_name('f1'), 'best_f1.pt')
        self.assertEqual(_monitor_checkpoint_name('val_loss'), 'best_val_loss.pt')


if __name__ == '__main__':
    unittest.main()
