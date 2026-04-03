from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.pose.pose_io import load_pose_npz

COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_pose(frame: np.ndarray, pose: np.ndarray, conf_thr: float = 0.05) -> np.ndarray:
    # pose: [M, K, 3]
    out = frame.copy()
    colors = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 90), (220, 100, 255)]

    for m in range(pose.shape[0]):
        color = colors[m % len(colors)]
        kp = pose[m]
        for e0, e1 in COCO17_EDGES:
            if kp[e0, 2] < conf_thr or kp[e1, 2] < conf_thr:
                continue
            p0 = tuple(np.round(kp[e0, :2]).astype(int))
            p1 = tuple(np.round(kp[e1, :2]).astype(int))
            cv2.line(out, p0, p1, color, 2)
        for j in range(kp.shape[0]):
            if kp[j, 2] < conf_thr:
                continue
            p = tuple(np.round(kp[j, :2]).astype(int))
            cv2.circle(out, p, 2, color, -1)
    return out


def render_overlay_video(video_path: str, pose_path: str, out_path: str, fps: float = 30.0) -> None:
    pose = load_pose_npz(pose_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps > 0:
        fps = src_fps

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < pose.shape[0]:
            frame = draw_pose(frame, pose[idx])
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
