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
from scene_decoupling.src.data.dataset import VideoPoseContextDataset
from scene_decoupling.src.engine.inference import infer_batch
from scene_decoupling.src.models.context_model import ContextDecoupledMemoryModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Inference for one video+pose pair (scene decoupling)')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--pose', required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--output', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])

    rec = {
        'sample_id': f'infer/{Path(args.video).stem}',
        'video_id': Path(args.video).stem,
        'split': 'infer',
        'class_name': 'unknown',
        'label': 0,
        'video_path': str(Path(args.video).resolve()),
        'pose_path': str(Path(args.pose).resolve()),
    }

    ds = VideoPoseContextDataset([rec], cfg['data'], split='val', seed=int(cfg['project']['seed']))
    sample = ds[0]

    clips = sample['clips'].unsqueeze(0)
    poses = sample['poses'].unsqueeze(0)
    clip_valid_mask = torch.ones((1, clips.shape[1]), dtype=torch.bool)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContextDecoupledMemoryModel(cfg['model'], cfg['data']).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    amp_dtype = torch.bfloat16 if cfg['runtime'].get('amp_dtype', 'bf16') == 'bf16' else torch.float16
    out = infer_batch(
        model=model,
        clips=clips,
        poses=poses,
        clip_valid_mask=clip_valid_mask,
        device=device,
        use_amp=bool(cfg['runtime'].get('use_amp', True)),
        amp_dtype=amp_dtype,
    )

    payload = {
        'video_id': sample['video_id'],
        'video_prob': float(out['video_prob'].squeeze().item()),
        'fg_ratio': float(out['fg_ratio'].squeeze().item()),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))

    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
