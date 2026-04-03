from __future__ import annotations

import sys
import unittest
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scene_decoupling.src.common.config import load_config, resolve_paths


HAS_CV2 = importlib.util.find_spec('cv2') is not None


@unittest.skipUnless(HAS_CV2, 'opencv-python is required for dataset decode tests')
class TestDataShapes(unittest.TestCase):
    def test_shape_and_padding(self) -> None:
        from scene_decoupling.src.data.builders import build_dataloaders, build_datasets

        cfg = resolve_paths(
            load_config(
                [
                    ROOT / 'scene_decoupling/configs/base.yaml',
                    ROOT / 'scene_decoupling/configs/data/rwf2000_video_pose.yaml',
                    ROOT / 'scene_decoupling/configs/data/rwf2000_curated_v1.yaml',
                    ROOT / 'scene_decoupling/configs/model/context_decoupled_mem.yaml',
                    ROOT / 'scene_decoupling/configs/train/default.yaml',
                    ROOT / 'scene_decoupling/configs/exp/rwf2000_scene_decoupling.yaml',
                ],
                overrides=[
                    'train.num_workers=0',
                    'train.persistent_workers=false',
                    'data.cache_mode=none',
                    'train.batch_size=2',
                    'data.max_clips_train=16',
                ],
            ),
            project_root=ROOT,
        )
        train_ds, val_ds = build_datasets(cfg)
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)

        item = train_ds[0]
        self.assertEqual(item['clips'].ndim, 5)  # [Nc, L, C, H, W]
        self.assertEqual(item['poses'].ndim, 5)  # [Nc, L, M, K, 3]
        h, w = cfg['data']['frame_size']
        self.assertGreaterEqual(float(item['poses'][..., 0].min().item()), 0.0)
        self.assertLessEqual(float(item['poses'][..., 0].max().item()), float(w - 1))
        self.assertGreaterEqual(float(item['poses'][..., 1].min().item()), 0.0)
        self.assertLessEqual(float(item['poses'][..., 1].max().item()), float(h - 1))

        train_ds.set_epoch(1)
        clips_e1 = train_ds[0]['clips']
        train_ds.set_epoch(2)
        clips_e2 = train_ds[0]['clips']
        self.assertGreater(float((clips_e1 - clips_e2).abs().mean().item()), 0.0)

        train_loader, _ = build_dataloaders(cfg, train_ds, val_ds, distributed=False)
        batch = next(iter(train_loader))
        self.assertEqual(batch['clips'].ndim, 6)
        self.assertEqual(batch['poses'].ndim, 6)
        self.assertEqual(batch['clip_valid_mask'].ndim, 2)


if __name__ == '__main__':
    unittest.main()
