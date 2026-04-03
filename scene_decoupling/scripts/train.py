#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.common.distributed import cleanup_distributed, init_distributed, is_main_process
from scene_decoupling.src.common.env import ensure_dirs, runtime_summary
from scene_decoupling.src.common.logger import build_logger
from scene_decoupling.src.common.seed import set_seed
from scene_decoupling.src.data.builders import build_dataloaders, build_datasets
from scene_decoupling.src.engine.trainer import build_optimizer, build_scheduler, fit
from scene_decoupling.src.exp.run_manager import RunManager
from scene_decoupling.src.models.context_model import ContextDecoupledMemoryModel
from scene_decoupling.src.models.losses import VideoLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train the scene-decoupling context model')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])

    dist_env = init_distributed(backend=cfg['train']['ddp']['backend'])

    if cfg['runtime'].get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg['runtime'].get('allow_tf32', True))
        torch.backends.cudnn.allow_tf32 = bool(cfg['runtime'].get('allow_tf32', True))
    torch.set_float32_matmul_precision(str(cfg['runtime'].get('matmul_precision', 'high')))

    set_seed(int(cfg['project']['seed']) + dist_env.rank, deterministic=bool(cfg['project'].get('deterministic', False)))

    ensure_dirs(cfg['paths']['output_root'], cfg['paths']['cache_root'])
    run_manager = RunManager(cfg['paths']['output_root'], cfg.get('experiment', {}).get('name', 'scene_decoupling'))

    logger = build_logger(
        'scene-decoupling-train',
        level=cfg['logging']['level'],
        log_file=str(run_manager.paths.run_dir / 'train.log') if is_main_process() else None,
    )

    if is_main_process():
        run_manager.dump_config(cfg)
        logger.info('Runtime summary: %s', runtime_summary())

    device = torch.device('cuda', dist_env.local_rank) if torch.cuda.is_available() else torch.device('cpu')

    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_ds, val_ds, distributed=dist_env.enabled)

    model = ContextDecoupledMemoryModel(cfg['model'], cfg['data']).to(device)
    streaming_backbone = bool(getattr(model.backbone, 'streaming_mode', False))
    if cfg['runtime'].get('compile_model', False):
        model = torch.compile(model)

    if dist_env.enabled and bool(cfg['train']['ddp'].get('enabled', True)):
        find_unused_parameters = bool(cfg['train']['ddp'].get('find_unused_parameters', False))
        # Official MeMViT online-memory blocks can have dynamic branch usage across
        # batches (e.g. very short clips), which requires unused-parameter handling in DDP.
        if streaming_backbone and not find_unused_parameters:
            if is_main_process():
                logger.warning(
                    'Detected streaming MeMViT backbone; forcing DDP find_unused_parameters=True '
                    'to avoid dynamic-graph reduction errors.'
                )
            find_unused_parameters = True
        model = DDP(
            model,
            device_ids=[dist_env.local_rank] if device.type == 'cuda' else None,
            find_unused_parameters=find_unused_parameters,
            static_graph=bool(cfg['train']['ddp'].get('static_graph', False)),
            gradient_as_bucket_view=bool(cfg['train']['ddp'].get('gradient_as_bucket_view', True)),
        )

    criterion = VideoLoss(
        pos_weight=float(cfg['loss'].get('pos_weight', 1.0)),
        focal_gamma=float(cfg['loss'].get('focal_gamma', 0.0)),
        label_smoothing=float(cfg['loss'].get('label_smoothing', 0.0)),
        sep_weight=float(cfg['loss'].get('sep_weight', 0.0)),
        overlap_weight=float(cfg['loss'].get('overlap_weight', 0.0)),
        fg_ratio_weight=float(cfg['loss'].get('fg_ratio_weight', 0.0)),
        fg_ratio_min=float(cfg['loss'].get('fg_ratio_min', 0.0)),
        fg_ratio_max=float(cfg['loss'].get('fg_ratio_max', 1.0)),
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    artifacts = fit(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        run_manager=run_manager,
        cfg=cfg,
        device=device,
        logger=logger,
    )

    if is_main_process():
        monitor = str(cfg['train']['early_stop'].get('monitor', 'val_loss'))
        logger.info(
            'Training done | monitor=%s best=%.4f | best_thr=%.3f | best_ckpt=%s',
            monitor,
            artifacts.best_metric,
            artifacts.best_threshold,
            artifacts.best_ckpt,
        )

    cleanup_distributed()


if __name__ == '__main__':
    main()
