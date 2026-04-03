#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter
from typing import Any

import numpy as np

from src.pose.pose_io import save_pose_npz
from src.pose.rtmpose_adapter import RTMPoseConfig, RTMPoseExtractor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_POSE_MODEL_FAST = "rtmpose-l_8xb256-420e_coco-256x192"
DEFAULT_POSE_MODEL_HQ = "rtmpose-x_8xb256-700e_body8-halpe26-384x288"
DEFAULT_DET_MODEL_FAST = "rtmdet_m_8xb32-300e_coco"
DEFAULT_DET_MODEL_HQ = "rtmdet_x_8xb32-300e_coco"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare RLVS for scene_decoupling (split + optional pose + index).")
    p.add_argument("--input-root", default="./data/rlvs")
    p.add_argument("--pose-root", default="./data/rlvs_pose_hq")
    p.add_argument("--index-file", default="scene_decoupling/outputs/cache/rlvs_scene_decoupling_index.jsonl")
    p.add_argument("--report", default="scene_decoupling/outputs/cache/rlvs_scene_decoupling_prepare_report.json")
    p.add_argument("--classes", default="Violence,NonViolence")
    p.add_argument("--class-map", default="Violence:1,NonViolence:0")
    p.add_argument("--video-exts", default="mp4,avi,mov,mkv")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--limit", type=int, default=0, help="Use first N videos after split assignment (debug only).")
    p.add_argument("--num-shards", type=int, default=1, help="Number of extraction shards for multi-GPU parallelism.")
    p.add_argument("--shard-id", type=int, default=0, help="Current shard id in [0, num-shards).")
    p.add_argument("--key-name", default="keypoints")
    p.add_argument("--max-persons", type=int, default=5)
    p.add_argument("--num-keypoints", type=int, default=17)

    p.add_argument("--extract-pose", action="store_true", help="Run RTMPose extraction for missing pose files.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing pose files.")
    p.add_argument("--allow-missing-pose", action="store_true", help="Write index with existing pose files only.")
    p.add_argument("--dry-run", action="store_true", help="Only show split stats, do not build index.")
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract/check pose files for the selected shard; do not write index.",
    )
    p.add_argument(
        "--build-index-only",
        action="store_true",
        help="Only build index from existing pose files; do not run extraction.",
    )

    p.add_argument("--pose-model", default="")
    p.add_argument("--det-model", default="")
    p.add_argument("--det-weights", default="")
    p.add_argument("--det-cat-ids", default="0")
    p.add_argument("--device", default="")
    p.add_argument("--high-precision", action="store_true")
    p.add_argument("--infer-batch-size", type=int, default=1)
    p.add_argument("--bbox-thr", type=float, default=0.3)
    p.add_argument("--nms-thr", type=float, default=0.3)
    return p.parse_args()


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_class_map(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in parse_csv(text):
        if ":" not in item:
            raise ValueError(f"Invalid class-map entry: {item}. Expected format name:label")
        k, v = item.split(":", 1)
        out[k.strip()] = int(v.strip())
    if not out:
        raise ValueError("class-map must not be empty")
    return out


def parse_det_cat_ids(text: str) -> tuple[int, ...] | None:
    s = (text or "").strip()
    if not s:
        return None
    vals: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return tuple(vals) if vals else None


def collect_videos(input_root: Path, classes: list[str], exts: set[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for cls in classes:
        class_dir = input_root / cls
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in exts]
        out[cls] = sorted(files)
    return out


def stratified_split(videos: dict[str, list[Path]], val_ratio: float, seed: int) -> list[tuple[str, str, Path]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val-ratio must be in (0,1), got {val_ratio}")
    rng = np.random.default_rng(int(seed))
    rows: list[tuple[str, str, Path]] = []
    for cls, files in videos.items():
        n = len(files)
        if n == 0:
            continue
        idx = np.arange(n, dtype=np.int64)
        rng.shuffle(idx)
        n_val = int(round(n * float(val_ratio)))
        if n >= 2:
            n_val = max(1, min(n - 1, n_val))
        else:
            n_val = 0
        val_idx = set(idx[:n_val].tolist())
        for i, fp in enumerate(files):
            split = "val" if i in val_idx else "train"
            rows.append((split, cls, fp))
    rows.sort(key=lambda x: (x[0], x[1], x[2].name))
    return rows


def npz_name_for(video_path: Path) -> str:
    # Keep extension in filename to avoid collisions when both .avi and .mp4 exist.
    return f"{video_path.stem}__{video_path.suffix.lower().lstrip('.')}.npz"


def inspect_pose(npz_path: Path, key_name: str) -> dict[str, int]:
    arr = np.load(npz_path)[key_name]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D pose in {npz_path}, got {arr.shape}")
    t, m, k, c = arr.shape
    return {
        "pose_frames": int(t),
        "pose_persons": int(m),
        "pose_joints": int(k),
        "pose_channels": int(c),
    }


def main() -> None:
    args = parse_args()
    if args.extract_only and args.build_index_only:
        raise ValueError("--extract-only and --build-index-only cannot be enabled at the same time.")
    if int(args.num_shards) <= 0:
        raise ValueError(f"--num-shards must be > 0, got {args.num_shards}")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError(f"--shard-id must be in [0, {int(args.num_shards) - 1}], got {args.shard_id}")
    if args.build_index_only and int(args.num_shards) != 1:
        raise ValueError("--build-index-only requires --num-shards=1")

    input_root = Path(args.input_root).resolve()
    pose_root = Path(args.pose_root).resolve()
    index_file = Path(args.index_file).resolve()
    report_file = Path(args.report).resolve()

    classes = parse_csv(args.classes)
    class_map = parse_class_map(args.class_map)
    for cls in classes:
        if cls not in class_map:
            raise ValueError(f"class-map missing class: {cls}")
    exts = {x.lower() for x in parse_csv(args.video_exts)}
    videos = collect_videos(input_root=input_root, classes=classes, exts=exts)
    assigned_all = stratified_split(videos=videos, val_ratio=float(args.val_ratio), seed=int(args.seed))
    if args.limit > 0:
        assigned_all = assigned_all[: int(args.limit)]
    assigned = [x for i, x in enumerate(assigned_all) if i % int(args.num_shards) == int(args.shard_id)]

    split_dist_all = Counter((split, cls) for split, cls, _ in assigned_all)
    split_dist_shard = Counter((split, cls) for split, cls, _ in assigned)
    payload: dict[str, Any] = {
        "input_root": str(input_root),
        "pose_root": str(pose_root),
        "index_file": str(index_file),
        "num_videos_selected_all": int(len(assigned_all)),
        "num_videos_selected_shard": int(len(assigned)),
        "class_distribution": {cls: int(len(videos.get(cls, []))) for cls in classes},
        "split_distribution_all": {f"{k[0]}/{k[1]}": int(v) for k, v in split_dist_all.items()},
        "split_distribution_shard": {f"{k[0]}/{k[1]}": int(v) for k, v in split_dist_shard.items()},
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
    }

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return

    extractor = None
    if args.extract_pose and not args.build_index_only:
        pose_model = args.pose_model or (DEFAULT_POSE_MODEL_HQ if args.high_precision else DEFAULT_POSE_MODEL_FAST)
        det_model = args.det_model or (DEFAULT_DET_MODEL_HQ if args.high_precision else DEFAULT_DET_MODEL_FAST)
        extractor = RTMPoseExtractor(
            RTMPoseConfig(
                pose_model=pose_model,
                det_model=det_model,
                max_persons=int(args.max_persons),
                num_keypoints=int(args.num_keypoints),
                device=(args.device.strip() or None),
                det_weights=(args.det_weights.strip() or None),
                det_cat_ids=parse_det_cat_ids(args.det_cat_ids),
                infer_batch_size=int(args.infer_batch_size),
                bbox_thr=float(args.bbox_thr),
                nms_thr=float(args.nms_thr),
            )
        )
        payload["extract_pose"] = {
            "pose_model": pose_model,
            "det_model": det_model,
            "det_weights": (args.det_weights.strip() or ""),
            "det_cat_ids": list(parse_det_cat_ids(args.det_cat_ids) or []),
            "device": (args.device.strip() or "auto"),
            "infer_batch_size": int(args.infer_batch_size),
            "bbox_thr": float(args.bbox_thr),
            "nms_thr": float(args.nms_thr),
        }

    processed = 0
    skipped = 0
    failed = 0
    errors: list[dict[str, str]] = []
    missing_pose: list[str] = []
    rows: list[dict[str, Any]] = []

    iterator = assigned
    if tqdm is not None:
        iterator = tqdm(assigned, total=len(assigned), desc="RLVS-Prep", leave=False)

    for split, cls, video_path in iterator:
        out_npz = pose_root / split / cls / npz_name_for(video_path)
        out_npz.parent.mkdir(parents=True, exist_ok=True)

        if extractor is not None and (args.overwrite or not out_npz.exists()):
            try:
                arr = extractor.extract_video(str(video_path))
                save_pose_npz(str(out_npz), arr)
                processed += 1
            except Exception as exc:  # pragma: no cover
                failed += 1
                errors.append({"video": str(video_path), "error": str(exc)})
                continue
        elif out_npz.exists():
            skipped += 1

        if not out_npz.exists():
            missing_pose.append(str(out_npz))
            continue

        meta = inspect_pose(out_npz, key_name=args.key_name)
        rows.append(
            {
                "sample_id": f"{split}/{cls}/{video_path.stem}",
                "video_id": video_path.stem,
                "split": split,
                "class_name": cls,
                "label": int(class_map[cls]),
                "video_path": str(video_path.resolve()),
                "pose_path": str(out_npz.resolve()),
                **meta,
            }
        )

    if missing_pose and not args.allow_missing_pose:
        payload["status"] = "failed_missing_pose"
        payload["num_missing_pose"] = int(len(missing_pose))
        payload["missing_pose_examples"] = missing_pose[:20]
        payload["num_failed_extract"] = int(failed)
        payload["extract_errors"] = errors[:20]
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Missing pose files detected. Re-run with --extract-pose or use --allow-missing-pose to continue."
        )

    payload.update(
        {
            "status": "ok_extract_only" if args.extract_only else "ok",
            "num_index_rows": int(len(rows)),
            "num_pose_processed": int(processed),
            "num_pose_skipped": int(skipped),
            "num_pose_failed": int(failed),
            "num_missing_pose": int(len(missing_pose)),
            "missing_pose_examples": missing_pose[:20],
            "extract_errors": errors[:20],
        }
    )
    if not args.extract_only:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with index_file.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        row_split = Counter((x["split"], x["class_name"]) for x in rows)
        payload["index_distribution"] = {f"{k[0]}/{k[1]}": int(v) for k, v in row_split.items()}

    print(json.dumps(payload, ensure_ascii=True, indent=2))
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
