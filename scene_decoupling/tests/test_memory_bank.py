from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scene_decoupling.src.models.memory_bank import FIFOStreamMemory


class TestMemoryBank(unittest.TestCase):
    def test_fifo_limit(self) -> None:
        bank = FIFOStreamMemory(max_steps=3, stop_grad=True)
        for _ in range(5):
            bank.append(torch.randn(2, 4, 8))
        self.assertEqual(len(bank), 3)

    def test_gather_shape(self) -> None:
        bank = FIFOStreamMemory(max_steps=2, stop_grad=False)
        bank.append(torch.randn(1, 3, 16), valid_mask=torch.tensor([True]))
        kv, pad = bank.gather(torch.randn(1, 2, 16), torch.tensor([True]))
        self.assertEqual(tuple(kv.shape), (1, 5, 16))
        self.assertEqual(tuple(pad.shape), (1, 5))


if __name__ == '__main__':
    unittest.main()
