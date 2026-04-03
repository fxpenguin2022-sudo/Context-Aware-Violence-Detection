from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scene_decoupling.src.models.mask import build_masks_by_mode, build_multiscale_mask_pyramid


class TestMaskModes(unittest.TestCase):
    def test_modes(self) -> None:
        poses = torch.zeros(2, 4, 3, 17, 3)
        poses[..., 0] = 40.0
        poses[..., 1] = 50.0
        poses[..., 2] = 0.8

        for mode in ['skeleton', 'random', 'none']:
            fg, bg = build_masks_by_mode(
                poses=poses,
                mode=mode,
                out_h=28,
                out_w=28,
                in_h=224,
                in_w=224,
                sigma=12.0,
                threshold=0.05,
            )
            self.assertEqual(tuple(fg.shape), (2, 4, 1, 28, 28))
            self.assertEqual(tuple(bg.shape), (2, 4, 1, 28, 28))

    def test_multiscale(self) -> None:
        fg = torch.rand(1, 3, 1, 28, 28)
        bg = 1.0 - fg
        fg_levels, bg_levels = build_multiscale_mask_pyramid(fg, bg, [(28, 28), (14, 14), (7, 7)])
        self.assertEqual(len(fg_levels), 3)
        self.assertEqual(tuple(fg_levels[1].shape), (1, 3, 1, 14, 14))
        self.assertEqual(tuple(bg_levels[2].shape), (1, 3, 1, 7, 7))


if __name__ == '__main__':
    unittest.main()
