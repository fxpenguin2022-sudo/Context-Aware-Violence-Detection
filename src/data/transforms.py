from __future__ import annotations

import numpy as np


def normalize_keypoints(
    x: np.ndarray,
    center_joint: int,
    eps: float = 1e-6,
) -> np.ndarray:
    # x: [T, M, K, 3] with (x, y, conf)
    out = x.copy()

    center = out[:, :, center_joint : center_joint + 1, :2]
    out[:, :, :, :2] = out[:, :, :, :2] - center

    # Torso-based scale from shoulders and hips where available.
    left_shoulder = out[:, :, 5, :2]
    right_shoulder = out[:, :, 6, :2]
    left_hip = out[:, :, 11, :2]
    right_hip = out[:, :, 12, :2]

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)
    hip_dist = np.linalg.norm(left_hip - right_hip, axis=-1)
    scale = np.maximum((shoulder_dist + hip_dist) * 0.5, eps)
    scale = scale[:, :, None, None]
    out[:, :, :, :2] = out[:, :, :, :2] / scale
    return out


def add_velocity_channel(x: np.ndarray) -> np.ndarray:
    # x: [T, M, K, 3] -> [T, M, K, 5]
    vel = np.zeros_like(x[..., :2])
    vel[1:] = x[1:, :, :, :2] - x[:-1, :, :, :2]
    return np.concatenate([x, vel], axis=-1)


def random_augment(
    x: np.ndarray,
    jitter_std: float,
    drop_joint_prob: float,
    drop_person_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = x.copy()

    if jitter_std > 0:
        noise = rng.normal(0.0, jitter_std, size=out[:, :, :, :2].shape).astype(np.float32)
        out[:, :, :, :2] += noise

    if drop_joint_prob > 0:
        joint_mask = rng.random(size=out[:, :, :, 0].shape) < drop_joint_prob  # [T, M, K]
        out[:, :, :, :2] = np.where(joint_mask[..., None], 0.0, out[:, :, :, :2])
        out[:, :, :, 2] = np.where(joint_mask, 0.0, out[:, :, :, 2])

    if drop_person_prob > 0:
        person_mask = rng.random(size=out[:, :, 0, 0].shape) < drop_person_prob  # [T, M]
        out = np.where(person_mask[:, :, None, None], 0.0, out)

    return out
