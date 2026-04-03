#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scene_decoupling.src.common.checkpoint import load_checkpoint
from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.data.dataset import VideoPoseContextDataset
from scene_decoupling.src.data.index_builder import load_index
from scene_decoupling.src.models.context_model import ContextDecoupledMemoryModel


_TIMES_FONT_PATH = fm.findfont('Times New Roman', fallback_to_default=True)
plt.rcParams.update(
    {
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman'],
        'axes.unicode_minus': False,
    }
)


def _resolve_record(args: argparse.Namespace, cfg: dict) -> dict[str, object]:
    if args.sample_id:
        index_file = args.index_file or str(cfg['paths']['index_file'])
        rows = load_index(index_file, split=args.split or None)
        for row in rows:
            if row['sample_id'] == args.sample_id:
                return row
        raise ValueError(f'Sample not found in index: {args.sample_id}')
    if not args.video or not args.pose:
        raise ValueError('Provide either --sample-id/--index-file or --video/--pose')
    return {
        'sample_id': f'vis/{Path(args.video).stem}',
        'video_id': Path(args.video).stem,
        'split': 'vis',
        'class_name': 'unknown',
        'label': 0,
        'video_path': str(Path(args.video).resolve()),
        'pose_path': str(Path(args.pose).resolve()),
    }


def _build_sample(cfg: dict, rec: dict[str, object]) -> dict[str, object]:
    ds = VideoPoseContextDataset([rec], cfg['data'], split='val', seed=int(cfg['project']['seed']))
    return ds[0]


def _to_cpu_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return np.sum(a_n * b_n, axis=1)


def _cosine_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if y is None:
        y = x
    x_n = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    y_n = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-8, None)
    return x_n @ y_n.T


def _pca_2d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.concatenate([a, b], axis=0)
    z = z - z.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    coords = u[:, :2] * s[:2]
    ratio = (s[:2] ** 2) / max(np.sum(s**2), 1e-8)
    return coords[: len(a)], coords[len(a) :], ratio


def _denorm_frame(frame_chw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.transpose(frame_chw, (1, 2, 0))
    x = x * std + mean
    x = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
    return x


def _save_clip_strip(path: Path, clips: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    if clips.shape[0] == 0:
        return
    clip_len = int(clips.shape[1])
    center = clip_len // 2
    tiles: list[np.ndarray] = []
    font = ImageFont.truetype(_TIMES_FONT_PATH, size=24)
    for idx in range(clips.shape[0]):
        frame = _denorm_frame(clips[idx, center], mean, std)
        img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(img)
        draw.text((10, 8), f'clip {idx}', font=font, fill=(255, 255, 0))
        tiles.append(np.asarray(img))
    strip = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def _plot_temporal(path: Path, clip_idx: np.ndarray, cos_as: np.ndarray, norm_a: np.ndarray, norm_s: np.ndarray, fg: np.ndarray, overlap: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(clip_idx, cos_as, marker='o', color='#c23b22')
    axes[0].axhline(0.0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].set_yticks(np.linspace(-1.0, 1.0, 5))
    axes[0].set_ylabel('cos(action, scene)')
    axes[0].set_title('Temporal Stream Separation')

    axes[1].plot(clip_idx, norm_a, marker='o', label='action norm', color='#0b6e4f')
    axes[1].plot(clip_idx, norm_s, marker='s', label='scene norm', color='#355c7d')
    axes[1].legend()
    axes[1].set_ylabel('feature norm')

    axes[2].plot(clip_idx, fg, marker='o', label='fg ratio', color='#f4a259')
    axes[2].plot(clip_idx, overlap, marker='s', label='mask overlap', color='#5b5f97')
    axes[2].legend()
    axes[2].set_ylabel('mask stats')
    axes[2].set_xlabel('clip index')

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scatter(path: Path, action_xy: np.ndarray, scene_xy: np.ndarray, cos_as: np.ndarray, expl_var: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(action_xy)))
    for idx, color in enumerate(colors):
        action_label = 'Action stream' if idx == 0 else None
        scene_label = 'Scene stream' if idx == 0 else None
        ax.scatter(action_xy[idx, 0], action_xy[idx, 1], color=color, marker='o', s=55, label=action_label)
        ax.scatter(scene_xy[idx, 0], scene_xy[idx, 1], color=color, marker='^', s=55, label=scene_label)
        ax.plot([action_xy[idx, 0], scene_xy[idx, 0]], [action_xy[idx, 1], scene_xy[idx, 1]], color=color, alpha=0.45)
    ax.set_title(
        'PCA of Action/Scene Step Features\n'
        f'PC1={expl_var[0]:.3f}, PC2={expl_var[1]:.3f}, mean cos={float(np.mean(cos_as)):.3f}'
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_similarity(path: Path, sim_action: np.ndarray, sim_scene: np.ndarray, sim_cross: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    mats = [sim_action, sim_scene, sim_cross]
    titles = ['Action-Action', 'Scene-Scene', 'Action-Scene']
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap='coolwarm')
        ax.set_title(title)
        ax.set_xlabel('clip index')
        ax.set_ylabel('clip index')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize checkpoint-dependent action/scene stream features')
    parser.add_argument('--config', nargs='+', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--video', default='')
    parser.add_argument('--pose', default='')
    parser.add_argument('--sample-id', default='')
    parser.add_argument('--index-file', default='')
    parser.add_argument('--split', default='')
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])
    rec = _resolve_record(args, cfg)
    sample = _build_sample(cfg, rec)

    clips = sample['clips'].unsqueeze(0)
    poses = sample['poses'].unsqueeze(0)
    clip_valid_mask = torch.ones((1, clips.shape[1]), dtype=torch.bool)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContextDecoupledMemoryModel(cfg['model'], cfg['data']).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    use_amp = bool(cfg['runtime'].get('use_amp', True))
    amp_dtype = torch.bfloat16 if cfg['runtime'].get('amp_dtype', 'bf16') == 'bf16' else torch.float16
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            out = model(
                clips.to(device),
                poses.to(device),
                clip_valid_mask=clip_valid_mask.to(device),
                return_debug=True,
            )

    valid = _to_cpu_numpy(out['valid_steps'][0]).astype(bool)
    action_steps = _to_cpu_numpy(out['evol_steps'][0])[valid]
    scene_steps = _to_cpu_numpy(out['scene_steps'][0])[valid]
    fg_steps = _to_cpu_numpy(out['fg_ratio_steps'][0])[valid]
    overlap_steps = _to_cpu_numpy(out['mask_overlap_steps'][0])[valid]
    clip_idx = np.arange(action_steps.shape[0], dtype=np.int64)

    if action_steps.shape[0] == 0:
        raise RuntimeError('No valid step features produced for visualization.')

    cos_as = _cosine_rows(action_steps, scene_steps)
    norm_action = np.linalg.norm(action_steps, axis=1)
    norm_scene = np.linalg.norm(scene_steps, axis=1)
    sim_action = _cosine_matrix(action_steps)
    sim_scene = _cosine_matrix(scene_steps)
    sim_cross = _cosine_matrix(action_steps, scene_steps)
    action_xy, scene_xy, expl_var = _pca_2d(action_steps, scene_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = np.array(cfg['data']['normalize']['mean'], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg['data']['normalize']['std'], dtype=np.float32).reshape(1, 1, 3)
    _save_clip_strip(out_dir / 'clip_centers.png', sample['clips'].numpy(), mean, std)
    _plot_temporal(out_dir / 'temporal_metrics.png', clip_idx, cos_as, norm_action, norm_scene, fg_steps, overlap_steps)
    _plot_scatter(out_dir / 'stream_scatter.png', action_xy, scene_xy, cos_as, expl_var)
    _plot_similarity(out_dir / 'stream_similarity.png', sim_action, sim_scene, sim_cross)

    payload = {
        'sample_id': sample['sample_id'],
        'video_id': sample['video_id'],
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_epoch': int(ckpt.get('epoch', -1)),
        'video_prob': float(out['video_prob'][0].detach().float().cpu().item()),
        'fg_ratio': float(out['fg_ratio'][0].detach().float().cpu().item()),
        'mask_overlap': float(out['mask_overlap'][0].detach().float().cpu().item()),
        'global_cosine': float(torch.nn.functional.cosine_similarity(out['f_evol'], out['f_scene'], dim=-1)[0].detach().cpu().item()),
        'temporal_mean_cosine': float(np.mean(cos_as)),
        'temporal_min_cosine': float(np.min(cos_as)),
        'temporal_max_cosine': float(np.max(cos_as)),
        'num_valid_steps': int(action_steps.shape[0]),
        'pca_explained_variance': [float(expl_var[0]), float(expl_var[1])],
        'per_step': [
            {
                'clip_index': int(i),
                'cosine': float(cos_as[i]),
                'action_norm': float(norm_action[i]),
                'scene_norm': float(norm_scene[i]),
                'fg_ratio': float(fg_steps[i]),
                'mask_overlap': float(overlap_steps[i]),
                'action_xy': [float(action_xy[i, 0]), float(action_xy[i, 1])],
                'scene_xy': [float(scene_xy[i, 0]), float(scene_xy[i, 1])],
            }
            for i in range(action_steps.shape[0])
        ],
    }
    (out_dir / 'summary.json').write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    np.savez_compressed(
        out_dir / 'features.npz',
        action_steps=action_steps,
        scene_steps=scene_steps,
        fg_ratio_steps=fg_steps,
        mask_overlap_steps=overlap_steps,
        action_similarity=sim_action,
        scene_similarity=sim_scene,
        cross_similarity=sim_cross,
    )

    print(json.dumps({'output_dir': str(out_dir.resolve()), 'summary': str((out_dir / 'summary.json').resolve())}, ensure_ascii=True))


if __name__ == '__main__':
    main()
