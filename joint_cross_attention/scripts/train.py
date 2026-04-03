#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from scene_decoupling.src.common.checkpoint import load_checkpoint
from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.common.distributed import cleanup_distributed, init_distributed, is_main_process
from scene_decoupling.src.common.env import ensure_dirs, runtime_summary
from scene_decoupling.src.common.logger import build_logger
from scene_decoupling.src.common.seed import set_seed
from scene_decoupling.src.exp.run_manager import RunManager
from joint_cross_attention.src.data.builders import build_dataloaders, build_datasets
from joint_cross_attention.src.engine.trainer import build_optimizer, build_scheduler, fit
from joint_cross_attention.src.models.joint_model import JointViolenceModel
from joint_cross_attention.src.models.losses import JointBCELoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train the joint model')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    return p.parse_args()


def _write_final_hparams(run_manager: RunManager, cfg: dict) -> None:
    payload = {
        'data': cfg['data'],
        'model': cfg['model'],
        'loss': cfg['loss'],
        'train': cfg['train'],
        'eval': cfg['eval'],
    }
    out = run_manager.paths.run_dir / 'final_hparams.json'
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')


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
    run_manager = RunManager(cfg['paths']['output_root'], cfg.get('experiment', {}).get('name', 'joint_model'))
    logger = build_logger('joint-model-train', level=cfg['logging']['level'], log_file=str(run_manager.paths.run_dir / 'train.log') if is_main_process() else None)
    if is_main_process():
        run_manager.dump_config(cfg)
        _write_final_hparams(run_manager, cfg)
        logger.info('Runtime summary: %s', runtime_summary())

    device = torch.device('cuda', dist_env.local_rank) if torch.cuda.is_available() else torch.device('cpu')
    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_ds, val_ds, distributed=dist_env.enabled)

    model = JointViolenceModel(cfg['model'], cfg['data']).to(device)
    warm_start_ckpt = str(cfg['model'].get('init_checkpoint', '')).strip()
    if warm_start_ckpt:
        payload = load_checkpoint(warm_start_ckpt, model, map_location='cpu')
        warm_epoch = payload.get('epoch', 'n/a')
        warm_metric = payload.get('best_metric', 'n/a')
        del payload
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if is_main_process():
            logger.info(
                'Warm-start joint model from %s | epoch=%s | best_metric=%s',
                warm_start_ckpt,
                warm_epoch,
                warm_metric,
            )
    if cfg['runtime'].get('compile_model', False):
        model = torch.compile(model)

    if dist_env.enabled and bool(cfg['train']['ddp'].get('enabled', True)):
        model = DDP(
            model,
            device_ids=[dist_env.local_rank] if device.type == 'cuda' else None,
            find_unused_parameters=bool(cfg['train']['ddp'].get('find_unused_parameters', True)),
            static_graph=bool(cfg['train']['ddp'].get('static_graph', False)),
            gradient_as_bucket_view=bool(cfg['train']['ddp'].get('gradient_as_bucket_view', True)),
        )

    criterion = JointBCELoss(pos_weight=float(cfg['loss'].get('pos_weight', 1.0)), label_smoothing=float(cfg['loss'].get('label_smoothing', 0.0))).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    artifacts = fit(model, criterion, train_loader, val_loader, optimizer, scheduler, run_manager, cfg, device, logger)

    if is_main_process():
        logger.info('Training done | monitor=%s best=%.4f | best_thr=%.3f | best_ckpt=%s', cfg['train']['early_stop'].get('monitor', 'acc'), artifacts.best_metric, artifacts.best_threshold, artifacts.best_ckpt)

    cleanup_distributed()


if __name__ == '__main__':
    main()
