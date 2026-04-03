from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.models.context_model import ContextDecoupledMemoryModel
from scene_decoupling.src.models.losses import VideoLoss


class TestForwardLoss(unittest.TestCase):
    def test_forward(self) -> None:
        cfg = resolve_paths(
            load_config(
                [
                    ROOT / 'scene_decoupling/configs/base.yaml',
                    ROOT / 'scene_decoupling/configs/data/rwf2000_video_pose.yaml',
                    ROOT / 'scene_decoupling/configs/model/context_decoupled_mem.yaml',
                    ROOT / 'scene_decoupling/configs/train/default.yaml',
                    ROOT / 'scene_decoupling/configs/exp/rwf2000_scene_decoupling.yaml',
                ],
                overrides=[
                    'data.frame_size=[112,112]',
                    'data.clip_len=4',
                    'model.backbone=resnet50_imagenet_pretrained',
                    'model.proj_dim=128',
                    'model.num_heads=4',
                    'model.attn_layers_per_stream=1',
                    'model.fusion.hidden_dim=128',
                ],
            ),
            project_root=ROOT,
        )

        b = 2
        nc = 6
        l = cfg['data']['clip_len']
        h, w = cfg['data']['frame_size']
        m = cfg['data']['max_persons']
        k = cfg['data']['num_keypoints']

        clips = torch.randn(b, nc, l, 3, h, w)
        poses = torch.randn(b, nc, l, m, k, 3)
        poses[..., 2] = torch.sigmoid(poses[..., 2])
        clip_valid_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        model = ContextDecoupledMemoryModel(cfg['model'], cfg['data'])
        out = model(clips, poses, clip_valid_mask=clip_valid_mask)
        self.assertEqual(tuple(out['video_logit'].shape), (b,))
        self.assertEqual(tuple(out['f_context'].shape), (b, cfg['model']['proj_dim']))

        y = torch.tensor([0.0, 1.0])
        loss = VideoLoss(pos_weight=1.0, focal_gamma=1.5)(out['video_logit'], y)['loss']
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
