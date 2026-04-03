from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.common.config import load_config, resolve_paths
from src.data.builders import build_datasets


class TestDataShapes(unittest.TestCase):
    def test_data_shapes(self) -> None:
        cfg = resolve_paths(
            load_config(
                [
                    ROOT / "configs/base.yaml",
                    ROOT / "configs/data/rwf2000_pose_hq.yaml",
                    ROOT / "configs/model/violence_skateformer_lite.yaml",
                    ROOT / "configs/train/ddp_amp.yaml",
                    ROOT / "configs/exp/baseline_mil.yaml",
                ]
            ),
            project_root=ROOT,
        )

        train_ds, _ = build_datasets(cfg)
        if len(train_ds) == 0:
            self.skipTest("No local pose files found for the default RWF-2000 paths.")
        item = train_ds[0]
        windows = item["windows"]
        self.assertEqual(windows.ndim, 5)
        self.assertEqual(windows.shape[2], cfg["data"]["max_persons"])
        self.assertEqual(windows.shape[3], cfg["data"]["num_keypoints"])


if __name__ == "__main__":
    unittest.main()
