#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse

import cv2
import numpy as np
import torch

from scene_decoupling.src.models.mask import build_skeleton_masks


def load_video(path: str, max_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description='Visualize skeleton-guided masks overlay')
    p.add_argument('--video', required=True)
    p.add_argument('--pose', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-frames', type=int, default=64)
    p.add_argument('--sigma', type=float, default=10.0)
    p.add_argument('--threshold', type=float, default=0.06)
    args = p.parse_args()

    video = load_video(args.video, args.max_frames)
    pose = np.load(args.pose)['keypoints'][: len(video)]

    h, w = video.shape[1], video.shape[2]
    poses = torch.from_numpy(pose[None, :, :, :, :3]).float()  # [1,T,M,K,3]
    fg, _ = build_skeleton_masks(poses, out_h=h, out_w=w, in_h=h, in_w=w, sigma=args.sigma, threshold=args.threshold)
    fg = fg.squeeze(0).squeeze(1).numpy()  # [T,H,W]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (w, h))
    for i in range(len(video)):
        frame = video[i].copy()
        mask = (fg[i] > 0).astype(np.uint8) * 255
        color_mask = np.zeros_like(frame)
        color_mask[..., 1] = mask
        blend = (0.7 * frame + 0.3 * color_mask).astype(np.uint8)
        writer.write(cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == '__main__':
    main()
