#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scene_decoupling.src.models.mask import build_masks_by_mode


def _parse_hw_list(text: str, fallback_h: int, fallback_w: int) -> list[tuple[int, int]]:
    raw = str(text).strip()
    if not raw:
        return [
            (max(1, fallback_h // 4), max(1, fallback_w // 4)),
            (max(1, fallback_h // 8), max(1, fallback_w // 8)),
            (max(1, fallback_h // 16), max(1, fallback_w // 16)),
        ]
    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if "x" not in t:
            raise ValueError(f"Invalid stage size token: {token}. Expect HxW format.")
        hs, ws = t.split("x", 1)
        h = max(1, int(hs))
        w = max(1, int(ws))
        out.append((h, w))
    if not out:
        raise ValueError("No valid stage sizes parsed from --stage-sizes.")
    return out


def _load_video_rgb(path: str, max_frames: int, resize_hw: tuple[int, int] | None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_hw is not None:
            frame = cv2.resize(frame, (int(resize_hw[1]), int(resize_hw[0])), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def _write_video_rgb(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    h, w = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _pool_mask(mask: torch.Tensor, out_h: int, out_w: int, strategy: str) -> torch.Tensor:
    if strategy == "max":
        return F.adaptive_max_pool2d(mask, output_size=(out_h, out_w))
    if strategy == "avg":
        return F.adaptive_avg_pool2d(mask, output_size=(out_h, out_w))
    raise ValueError(f"Unsupported strategy in visualization: {strategy}")


def _save_stage_heatmap(path: Path, mask_2d: np.ndarray) -> None:
    x = np.clip(mask_2d, 0.0, 1.0)
    x_u8 = (x * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(x_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), color)


def _motion_heat(prev_rgb: np.ndarray, cur_rgb: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cur_gray = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = np.abs(cur_gray - prev_gray)
    diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=1.2, sigmaY=1.2)
    maxv = float(diff.max())
    if maxv > 1e-6:
        diff = diff / maxv
    heat = cv2.applyColorMap((np.clip(diff, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize skeleton-guided foreground/background separation")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--pose", required=True, help="Pose npz path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--key-name", default="keypoints", help="Pose key inside npz")
    parser.add_argument("--mask-mode", default="skeleton", choices=["skeleton", "random", "none"])
    parser.add_argument("--sigma", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--resize-h", type=int, default=0)
    parser.add_argument("--resize-w", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=8, help="Save one panel image every N frames")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--save-motion",
        action="store_true",
        help="Also save motion heat videos based on adjacent-frame difference.",
    )
    parser.add_argument(
        "--stage-sizes",
        default="",
        help="Comma-separated HxW, e.g. 56x56,28x28,14x14. Empty means auto scales.",
    )
    parser.add_argument(
        "--align-strategy",
        default="max",
        choices=["max", "avg"],
        help="Downsample strategy for stage overlap diagnostics.",
    )
    parser.add_argument(
        "--enforce-complement",
        action="store_true",
        help="Mimic training-time bg = 1 - fg after stage downsampling.",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    resize_hw = None
    if int(args.resize_h) > 0 and int(args.resize_w) > 0:
        resize_hw = (int(args.resize_h), int(args.resize_w))

    video = _load_video_rgb(args.video, max_frames=int(args.max_frames), resize_hw=resize_hw)
    pose_np = np.load(args.pose)[str(args.key_name)].astype(np.float32)

    t = min(int(video.shape[0]), int(pose_np.shape[0]))
    if t <= 0:
        raise RuntimeError("No overlapping frames between video and pose.")
    video = video[:t]
    pose_np = pose_np[:t, :, :, :3]

    h, w = int(video.shape[1]), int(video.shape[2])
    poses = torch.from_numpy(pose_np[None]).float()  # [1,T,M,K,3]
    fg, bg = build_masks_by_mode(
        poses=poses,
        mode=str(args.mask_mode),
        out_h=h,
        out_w=w,
        in_h=h,
        in_w=w,
        sigma=float(args.sigma),
        threshold=float(args.threshold),
    )
    fg = fg.squeeze(0).squeeze(1).cpu().numpy()  # [T,H,W]
    bg = bg.squeeze(0).squeeze(1).cpu().numpy()  # [T,H,W]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    overlay_fg_frames: list[np.ndarray] = []
    overlay_bg_frames: list[np.ndarray] = []
    fg_only_frames: list[np.ndarray] = []
    bg_only_frames: list[np.ndarray] = []
    panel_frames: list[np.ndarray] = []
    motion_frames: list[np.ndarray] = []
    fg_motion_frames: list[np.ndarray] = []
    motion_panel_frames: list[np.ndarray] = []

    fg_ratios: list[float] = []
    bg_ratios: list[float] = []
    overlap_ratios: list[float] = []

    for i in range(t):
        frame = video[i].astype(np.float32)
        mfg = np.clip(fg[i], 0.0, 1.0)
        mbg = np.clip(bg[i], 0.0, 1.0)

        fg_ratios.append(float(mfg.mean()))
        bg_ratios.append(float(mbg.mean()))
        overlap_ratios.append(float((mfg * mbg).mean()))

        mfg3 = mfg[..., None]
        mbg3 = mbg[..., None]
        fg_only = np.clip(frame * mfg3, 0.0, 255.0).astype(np.uint8)
        bg_only = np.clip(frame * mbg3, 0.0, 255.0).astype(np.uint8)

        fg_color = np.zeros_like(frame)
        fg_color[..., 1] = 255.0
        bg_color = np.zeros_like(frame)
        bg_color[..., 2] = 255.0
        overlay_fg = np.clip(frame * 0.72 + fg_color * (0.28 * mfg3), 0.0, 255.0).astype(np.uint8)
        overlay_bg = np.clip(frame * 0.72 + bg_color * (0.28 * mbg3), 0.0, 255.0).astype(np.uint8)

        row1 = np.concatenate([video[i], overlay_fg], axis=1)
        row2 = np.concatenate([fg_only, bg_only], axis=1)
        panel = np.concatenate([row1, row2], axis=0)

        overlay_fg_frames.append(overlay_fg)
        overlay_bg_frames.append(overlay_bg)
        fg_only_frames.append(fg_only)
        bg_only_frames.append(bg_only)
        panel_frames.append(panel)

        if bool(args.save_motion):
            if i == 0:
                motion = np.zeros_like(video[i])
            else:
                motion = _motion_heat(video[i - 1], video[i])
            fg_motion = np.clip(motion.astype(np.float32) * mfg3, 0.0, 255.0).astype(np.uint8)
            motion_panel = np.concatenate([video[i], motion, fg_only, fg_motion], axis=1)
            motion_frames.append(motion)
            fg_motion_frames.append(fg_motion)
            motion_panel_frames.append(motion_panel)

        stride = max(1, int(args.frame_stride))
        if i % stride == 0:
            cv2.imwrite(str(frames_dir / f"panel_{i:04d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    _write_video_rgb(out_dir / "overlay_fg.mp4", np.stack(overlay_fg_frames, axis=0), fps=float(args.fps))
    _write_video_rgb(out_dir / "overlay_bg.mp4", np.stack(overlay_bg_frames, axis=0), fps=float(args.fps))
    _write_video_rgb(out_dir / "fg_only.mp4", np.stack(fg_only_frames, axis=0), fps=float(args.fps))
    _write_video_rgb(out_dir / "bg_only.mp4", np.stack(bg_only_frames, axis=0), fps=float(args.fps))
    _write_video_rgb(out_dir / "panel_2x2.mp4", np.stack(panel_frames, axis=0), fps=float(args.fps))
    if bool(args.save_motion):
        _write_video_rgb(out_dir / "motion_only.mp4", np.stack(motion_frames, axis=0), fps=float(args.fps))
        _write_video_rgb(out_dir / "fg_motion_only.mp4", np.stack(fg_motion_frames, axis=0), fps=float(args.fps))
        _write_video_rgb(out_dir / "motion_panel_1x4.mp4", np.stack(motion_panel_frames, axis=0), fps=float(args.fps))

    stage_sizes = _parse_hw_list(args.stage_sizes, fallback_h=h, fallback_w=w)
    stage_dir = out_dir / "stage_masks"
    stage_dir.mkdir(parents=True, exist_ok=True)

    fg_t = torch.from_numpy(fg).float().unsqueeze(1)  # [T,1,H,W]
    bg_t = torch.from_numpy(bg).float().unsqueeze(1)  # [T,1,H,W]
    stage_stats: list[dict[str, float]] = []
    for sh, sw in stage_sizes:
        fg_lvl = _pool_mask(fg_t, out_h=sh, out_w=sw, strategy=str(args.align_strategy))
        if bool(args.enforce_complement):
            bg_lvl = (1.0 - fg_lvl).clamp(0.0, 1.0)
        else:
            bg_lvl = _pool_mask(bg_t, out_h=sh, out_w=sw, strategy=str(args.align_strategy))
        fg_mean = fg_lvl.mean(dim=0).squeeze(0).cpu().numpy()
        bg_mean = bg_lvl.mean(dim=0).squeeze(0).cpu().numpy()
        overlap = (fg_lvl * bg_lvl).mean().item()
        stage_stats.append(
            {
                "h": int(sh),
                "w": int(sw),
                "fg_mean": float(fg_lvl.mean().item()),
                "bg_mean": float(bg_lvl.mean().item()),
                "fg_bg_overlap": float(overlap),
            }
        )
        _save_stage_heatmap(stage_dir / f"fg_mean_{sh}x{sw}.png", fg_mean)
        _save_stage_heatmap(stage_dir / f"bg_mean_{sh}x{sw}.png", bg_mean)

    payload = {
        "video": str(Path(args.video).resolve()),
        "pose": str(Path(args.pose).resolve()),
        "mask_mode": str(args.mask_mode),
        "sigma": float(args.sigma),
        "threshold": float(args.threshold),
        "num_frames": int(t),
        "frame_h": int(h),
        "frame_w": int(w),
        "fg_ratio_mean": float(np.mean(fg_ratios)),
        "bg_ratio_mean": float(np.mean(bg_ratios)),
        "fg_bg_overlap_mean": float(np.mean(overlap_ratios)),
        "fg_ratio_per_frame": fg_ratios,
        "bg_ratio_per_frame": bg_ratios,
        "fg_bg_overlap_per_frame": overlap_ratios,
        "align_strategy": str(args.align_strategy),
        "enforce_complement": bool(args.enforce_complement),
        "stage_stats": stage_stats,
    }
    (out_dir / "stats.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir.resolve()), "stats": str((out_dir / "stats.json").resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
