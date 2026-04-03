#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse

import torch

from scene_decoupling.src.common.config import load_config, resolve_paths
from joint_cross_attention.src.data.builders import build_dataloaders, build_datasets
from joint_cross_attention.src.models.joint_model import JointViolenceModel
from joint_cross_attention.src.models.losses import JointBCELoss
from scene_decoupling.src.engine.metrics import binary_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run joint-model pipeline sanity checks')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])
    train_ds, val_ds = build_datasets(cfg)
    train_loader, _ = build_dataloaders(cfg, train_ds, val_ds, distributed=False)
    batch = next(iter(train_loader))

    print('[check] pose_windows shape', tuple(batch['pose_windows'].shape))
    print('[check] video_clips shape', tuple(batch['video_clips'].shape))
    print('[check] video_poses shape', tuple(batch['video_poses'].shape))
    print('[check] clip_valid_mask shape', tuple(batch['clip_valid_mask'].shape))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointViolenceModel(cfg['model'], cfg['data']).to(device)
    out = model(
        batch['pose_windows'].to(device).float(),
        batch['pose_window_valid'].to(device).bool(),
        batch['video_clips'].to(device).float(),
        batch['video_poses'].to(device).float(),
        batch['clip_valid_mask'].to(device).bool(),
    )
    print('[check] video_logit', tuple(out['video_logit'].shape), 'alpha mean', float(out['alpha'].mean().item()))
    assert torch.allclose(out['alpha'] + out['beta'] + out['gamma'], torch.ones_like(out['alpha']), atol=1e-5)

    criterion = JointBCELoss(pos_weight=float(cfg['loss'].get('pos_weight', 1.0)), label_smoothing=float(cfg['loss'].get('label_smoothing', 0.0))).to(device)
    loss_dict = criterion(out['video_logit'], batch['label'].to(device).float())
    print('[check] loss', float(loss_dict['loss'].item()))
    assert torch.isfinite(loss_dict['loss'])

    y_true = torch.tensor([0, 0, 1, 1], dtype=torch.float32).numpy()
    y_prob = torch.tensor([0.1, 0.4, 0.6, 0.9], dtype=torch.float32).numpy()
    metric = binary_metrics(y_true, y_prob, threshold=0.5)
    print('[check] metric', metric)
    print('[check] all passed')


if __name__ == '__main__':
    main()
