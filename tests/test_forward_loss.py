from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.common.config import load_config, resolve_paths
from src.models.losses import VideoBCELoss
from src.models.model import ViolenceMILModel


class TestForwardLoss(unittest.TestCase):
    def test_forward_and_loss(self) -> None:
        cfg = resolve_paths(
            load_config(
                [
                    ROOT / "configs/base.yaml",
                    ROOT / "configs/data/rwf2000_pose_hq.yaml",
                    ROOT / "configs/model/violence_skateformer_lite.yaml",
                    ROOT / "configs/train/ddp_amp.yaml",
                    ROOT / "configs/exp/baseline_mil.yaml",
                ],
                overrides=["model.depth=2", "train.batch_size=2"],
            ),
            project_root=ROOT,
        )

        b, w = 2, 4
        t = cfg["data"]["window_size"]
        m = cfg["data"]["max_persons"]
        k = cfg["data"]["num_keypoints"]
        c = cfg["model"]["input_dim"]

        x = torch.randn(b, w, t, m, k, c)
        valid = torch.ones(b, w, t, dtype=torch.bool)
        y = torch.tensor([0.0, 1.0])

        model = ViolenceMILModel(cfg["model"], cfg["data"])
        out = model(x, valid)

        crit = VideoBCELoss()
        loss = crit(out["video_logit"], y, out["window_probs"])["loss"]
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
