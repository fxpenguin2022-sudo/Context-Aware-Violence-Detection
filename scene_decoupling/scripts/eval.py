#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json

import torch

from scene_decoupling.src.common.checkpoint import load_checkpoint
from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.common.logger import build_logger
from scene_decoupling.src.data.builders import build_dataloaders, build_datasets
from scene_decoupling.src.engine.evaluator import evaluate
from scene_decoupling.src.models.context_model import ContextDecoupledMemoryModel
from scene_decoupling.src.models.losses import VideoLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate the scene-decoupling model')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--output', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])

    logger = build_logger('scene-decoupling-eval', level=cfg['logging']['level'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_ds = build_datasets(cfg)
    _, val_loader = build_dataloaders(cfg, val_ds, val_ds, distributed=False)

    model = ContextDecoupledMemoryModel(cfg['model'], cfg['data']).to(device)
    criterion = VideoLoss(
        pos_weight=float(cfg['loss'].get('pos_weight', 1.0)),
        focal_gamma=float(cfg['loss'].get('focal_gamma', 0.0)),
        sep_weight=float(cfg['loss'].get('sep_weight', 0.0)),
        overlap_weight=float(cfg['loss'].get('overlap_weight', 0.0)),
        fg_ratio_weight=float(cfg['loss'].get('fg_ratio_weight', 0.0)),
        fg_ratio_min=float(cfg['loss'].get('fg_ratio_min', 0.0)),
        fg_ratio_max=float(cfg['loss'].get('fg_ratio_max', 1.0)),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)

    fixed = float(cfg['eval']['threshold']['fixed'])
    if ckpt.get('extra') and ckpt['extra'].get('best_threshold') is not None:
        fixed = float(ckpt['extra']['best_threshold'])

    amp_dtype = torch.bfloat16 if cfg['runtime'].get('amp_dtype', 'bf16') == 'bf16' else torch.float16
    result = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        use_amp=bool(cfg['runtime'].get('use_amp', True)),
        amp_dtype=amp_dtype,
        fixed_threshold=fixed,
        scan_cfg=cfg['eval']['threshold'],
        criterion=criterion,
        loss_weight_scale=1.0,
        max_batches=int(cfg['eval'].get('max_batches', 0)),
        show_progress=bool(cfg['logging'].get('progress_bar', True)),
    )

    logger.info('Eval summary: %s', result.summary)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    'summary': result.summary,
                    'predictions': result.predictions,
                    'threshold_scan': result.threshold_records,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding='utf-8',
        )


if __name__ == '__main__':
    main()
