#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse

import numpy as np
import torch

from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.data.builders import build_dataloaders, build_datasets
from scene_decoupling.src.engine.metrics import binary_metrics
from scene_decoupling.src.models.context_model import ContextDecoupledMemoryModel
from scene_decoupling.src.models.losses import VideoLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Sanity checks for the scene-decoupling branch')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])

    train_ds, val_ds = build_datasets(cfg)
    train_loader, _ = build_dataloaders(cfg, train_ds, val_ds, distributed=False)

    batch = next(iter(train_loader))
    print('[check] clips shape', tuple(batch['clips'].shape))
    print('[check] poses shape', tuple(batch['poses'].shape))
    print('[check] clip_valid_mask shape', tuple(batch['clip_valid_mask'].shape))

    model = ContextDecoupledMemoryModel(cfg['model'], cfg['data'])
    out = model(batch['clips'].float(), batch['poses'].float(), clip_valid_mask=batch['clip_valid_mask'])
    print('[check] video_logit', tuple(out['video_logit'].shape), 'fg_ratio mean', float(out['fg_ratio'].mean().item()))

    crit = VideoLoss(pos_weight=float(cfg['loss'].get('pos_weight', 1.0)), focal_gamma=float(cfg['loss'].get('focal_gamma', 0.0)))
    loss = crit(out['video_logit'], batch['label'].float())['loss']
    print('[check] loss', float(loss.item()))
    assert torch.isfinite(loss)

    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    m = binary_metrics(y_true, y_prob, threshold=0.5)
    print('[check] metric', m)
    assert abs(m['f1'] - 1.0) < 1e-6

    print('[check] all passed')


if __name__ == '__main__':
    main()
