from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scene_decoupling.src.engine.evaluator import evaluate


class _DummyModel(torch.nn.Module):
    def forward(self, clips: torch.Tensor, poses: torch.Tensor, clip_valid_mask: torch.Tensor | None = None):
        logits = clips[:, 0, 0, 0, 0, 0]
        probs = torch.sigmoid(logits)
        fg_ratio = torch.full_like(probs, 0.25)
        return {
            'video_logit': logits,
            'video_prob': probs,
            'fg_ratio': fg_ratio,
        }


class TestEvaluatorSummary(unittest.TestCase):
    def test_summary_has_best_acc(self) -> None:
        batch = {
            'sample_id': ['a', 'b', 'c', 'd'],
            'video_id': ['a', 'b', 'c', 'd'],
            'video_path': ['va', 'vb', 'vc', 'vd'],
            'pose_path': ['pa', 'pb', 'pc', 'pd'],
            'label': torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32),
            'clips': torch.tensor([-2.0, 2.0, -0.5, 1.0], dtype=torch.float32).view(4, 1, 1, 1, 1, 1),
            'poses': torch.zeros((4, 1, 1, 1, 1, 3), dtype=torch.float32),
            'clip_valid_mask': torch.ones((4, 1), dtype=torch.bool),
        }

        result = evaluate(
            model=_DummyModel(),
            dataloader=[batch],
            device=torch.device('cpu'),
            use_amp=False,
            amp_dtype=torch.float32,
            fixed_threshold=0.5,
            scan_cfg={'scan_min': 0.01, 'scan_max': 0.99, 'scan_steps': 21},
            criterion=None,
            max_batches=0,
            show_progress=False,
        )

        self.assertIn('best', result.summary)
        self.assertIn('best_f1', result.summary)
        self.assertIn('best_acc', result.summary)
        self.assertIn('best_balanced_acc', result.summary)
        self.assertAlmostEqual(result.summary['best']['f1'], result.summary['best_f1']['f1'], places=6)


if __name__ == '__main__':
    unittest.main()
