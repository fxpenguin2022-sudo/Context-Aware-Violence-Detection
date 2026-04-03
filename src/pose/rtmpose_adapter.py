from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RTMPoseConfig:
    pose_model: str
    det_model: str
    max_persons: int
    num_keypoints: int
    device: str | None = None
    det_weights: str | None = None
    det_cat_ids: tuple[int, ...] | None = (0,)
    infer_batch_size: int = 1
    bbox_thr: float = 0.3
    nms_thr: float = 0.3


class RTMPoseExtractor:
    def __init__(self, cfg: RTMPoseConfig) -> None:
        self.cfg = cfg
        self._inferencer = None

    def _lazy_init(self) -> None:
        if self._inferencer is not None:
            return
        try:
            from mmpose.apis import MMPoseInferencer
            from mmpose.utils import register_all_modules as register_mmpose_modules
            from mmdet.utils import register_all_modules as register_mmdet_modules
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "RTMPose runtime dependencies are missing. Install mmpose/mmdet/mmcv/mmengine first."
            ) from exc

        # Explicitly register mmdet/mmpose scopes to avoid registry fallback warnings.
        register_mmdet_modules(init_default_scope=False)
        register_mmpose_modules(init_default_scope=True)

        self._inferencer = MMPoseInferencer(
            pose2d=self.cfg.pose_model,
            det_model=self.cfg.det_model,
            det_weights=self.cfg.det_weights,
            det_cat_ids=self.cfg.det_cat_ids,
            device=self.cfg.device,
            show_progress=False,
        )

    def _parse_predictions(self, pred: Any) -> np.ndarray:
        # Expected output is a list of person predictions.
        if isinstance(pred, dict) and "predictions" in pred:
            pred = pred["predictions"]
        if isinstance(pred, list) and pred and isinstance(pred[0], list):
            pred = pred[0]
        if pred is None:
            pred = []

        people = []
        for person in pred:
            kpts = None
            scores = None
            if isinstance(person, dict):
                kpts = person.get("keypoints", None)
                scores = person.get("keypoint_scores", None)
            elif hasattr(person, "pred_instances"):
                inst = person.pred_instances
                if hasattr(inst, "keypoints"):
                    kpts = np.asarray(inst.keypoints)
                if hasattr(inst, "keypoint_scores"):
                    scores = np.asarray(inst.keypoint_scores)

            if kpts is None:
                continue
            kpts = np.asarray(kpts, dtype=np.float32)
            if kpts.ndim == 3:
                kpts = kpts[0]

            if scores is None:
                scores = np.ones((kpts.shape[0],), dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            if scores.ndim > 1:
                scores = scores.reshape(-1)

            c = scores[:, None]
            if kpts.shape[0] != self.cfg.num_keypoints:
                if kpts.shape[0] > self.cfg.num_keypoints:
                    kpts = kpts[: self.cfg.num_keypoints]
                    c = c[: self.cfg.num_keypoints]
                else:
                    pad = self.cfg.num_keypoints - kpts.shape[0]
                    kpts = np.concatenate([kpts, np.zeros((pad, 2), dtype=np.float32)], axis=0)
                    c = np.concatenate([c, np.zeros((pad, 1), dtype=np.float32)], axis=0)

            people.append(np.concatenate([kpts[:, :2], c], axis=-1))

        if not people:
            return np.zeros((self.cfg.max_persons, self.cfg.num_keypoints, 3), dtype=np.float32)

        people_arr = np.stack(people, axis=0)
        conf = people_arr[..., 2].mean(axis=1)
        order = np.argsort(-conf)
        people_arr = people_arr[order]

        if people_arr.shape[0] >= self.cfg.max_persons:
            people_arr = people_arr[: self.cfg.max_persons]
        else:
            pad = np.zeros(
                (self.cfg.max_persons - people_arr.shape[0], self.cfg.num_keypoints, 3),
                dtype=np.float32,
            )
            people_arr = np.concatenate([people_arr, pad], axis=0)

        return people_arr

    def extract_video(self, video_path: str) -> np.ndarray:
        self._lazy_init()
        # Run inferencer once on the full video path to avoid repeated
        # per-frame Python-side setup overhead.
        result_iter = self._inferencer(
            video_path,
            return_vis=False,
            batch_size=max(1, int(self.cfg.infer_batch_size)),
            bbox_thr=float(self.cfg.bbox_thr),
            nms_thr=float(self.cfg.nms_thr),
        )
        frames = [self._parse_predictions(result) for result in result_iter]

        if not frames:
            return np.zeros((0, self.cfg.max_persons, self.cfg.num_keypoints, 3), dtype=np.float32)

        return np.stack(frames, axis=0)
